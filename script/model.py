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

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN INICIAL
# ──────────────────────────────────────────────────────────────
device = torch_directml.device()
print(f"✅ Usando dispositivo DirectML: {device}")

output_dir = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/NeoplasiaClassifier-ES/output"
os.makedirs(output_dir, exist_ok=True)

ruta_csv = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/Multiples_neoplasias_solo_resumenes_selection/textos_cortos_filtrados.csv"
modelo_path = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
batch_size = 8
epochs = 5
max_len = 512

# ──────────────────────────────────────────────────────────────
# CARGAR DATOS Y DIVIDIR EN TRAIN / VAL / TEST
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(ruta_csv)
df = df.rename(columns={"TEXTO": "texto", "MULTIPLES": "label"})

df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["label"], random_state=42)

# ──────────────────────────────────────────────────────────────
# DATASET PERSONALIZADO
# ──────────────────────────────────────────────────────────────
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

# Dataloaders
train_dataset = NeoplasiaDataset(df_train, tokenizer, max_len)
val_dataset = NeoplasiaDataset(df_val, tokenizer, max_len)
test_dataset = NeoplasiaDataset(df_test, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ──────────────────────────────────────────────────────────────
# MODELO Y ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(modelo_path, num_labels=2)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Archivo para guardar métricas por época
with open(f"{output_dir}/metricas_epocas.txt", "w") as f:
    f.write("Época\tTrain Loss\tVal Accuracy\tVal F1\n")

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
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_labels = torch.argmax(logits, dim=1)
            val_preds.extend(pred_labels.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)

    # Guardar métricas de la época
    with open(f"{output_dir}/metricas_epocas.txt", "a") as f:
        f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\n")

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

with open(f"{output_dir}/metricas_test.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")

# Reporte de clasificación completo
reporte = classification_report(test_labels, test_preds, digits=4)
with open(f"{output_dir}/reporte_clasificacion.txt", "w") as f:
    f.write(reporte)

# Matriz de confusión
cm = confusion_matrix(test_labels, test_preds)
plt.figure()
plt.matshow(cm, cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.colorbar()
plt.savefig(f"{output_dir}/matriz_confusion.png")
plt.close()

# ──────────────────────────────────────────────────────────────
# GUARDAR MODELO
# ──────────────────────────────────────────────────────────────
model.save_pretrained(f"{output_dir}/roberta_directml")
tokenizer.save_pretrained(f"{output_dir}/roberta_directml")
print(f"✅ Modelo guardado en: {output_dir}/roberta_directml")

