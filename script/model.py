import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch_directml
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


device = torch.device("cpu")

output_dir = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/NeoplasiaClassifier-ES/output/roberta/roberta3"
os.makedirs(output_dir, exist_ok=True)

ruta_csv = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/Multiples_neoplasias_solo_resumenes_selection/textos_cortos_filtrados.csv"
modelo_path = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
batch_size = 16
epochs = 15
max_len = 512
early_stopping_patience = 5


df = pd.read_csv(ruta_csv)
df = df.rename(columns={"TEXTO": "texto", "MULTIPLES": "label"})

df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["label"], random_state=42)


tokenizer = AutoTokenizer.from_pretrained(modelo_path)

class NeoplasiaDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        texto = str(self.df.iloc[idx]["texto"])
        label = int(self.df.iloc[idx]["label"])
        tokens = self.tokenizer(
            texto,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

train_dataset = NeoplasiaDataset(df_train, tokenizer, max_len)
val_dataset = NeoplasiaDataset(df_val, tokenizer, max_len)
test_dataset = NeoplasiaDataset(df_test, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


model = AutoModelForSequenceClassification.from_pretrained(modelo_path, num_labels=2)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

with open(f"{output_dir}/metricas.txt", "w") as f:
    f.write("Época\tTrain Loss\tVal Accuracy\tVal F1\n")

best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"🔁 Época {epoch+1}/{epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validación
    val_loss = 0.0  
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits
            pred_labels = torch.argmax(logits, dim=1)
            val_preds.extend(pred_labels.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            loss = outputs.loss  
            val_loss += loss.item()  


    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)
    val_loss /= len(val_loader)  


    with open(f"{output_dir}/metricas.txt", "a") as f:
        f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        model.save_pretrained(f"{output_dir}")
        tokenizer.save_pretrained(f"{output_dir}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print(f"⏹️ Early stopping en época {epoch+1}")
            break


# ──────────────────────────────────────────────────────────────
# EVALUACIÓN FINAL EN TEST
# ──────────────────────────────────────────────────────────────
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_labels = torch.argmax(logits, dim=1)
        test_preds.extend(pred_labels.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Métricas finales
acc = accuracy_score(test_labels, test_preds)
prec = precision_score(test_labels, test_preds)
rec = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
cm = confusion_matrix(test_labels, test_preds)


with open(f"{output_dir}/metricas.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write("\nMatriz de Confusión - TEST FINAL:\n")
    f.write(str(cm) + "\n")

plt.figure(figsize=(5, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - TEST")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Una neoplasia", "Múltiples"], rotation=45)
plt.yticks(tick_marks, ["Una neoplasia", "Múltiples"])


thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel("Real")
plt.xlabel("Predicción")
plt.tight_layout()
plt.savefig(f"{output_dir}/img/matriz_confusion.png")
plt.close()


model.save_pretrained(f"{output_dir}")
tokenizer.save_pretrained(f"{output_dir}")
print(f" Modelo guardado en: {output_dir}")

