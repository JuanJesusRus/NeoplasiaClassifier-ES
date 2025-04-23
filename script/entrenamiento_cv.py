import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class NeoplasiaDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
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



def cargarDatos(ruta_csv, modelo_path, max_len):

    df = pd.read_csv(ruta_csv)
    df = df.rename(columns={"TEXTO": "texto", "MULTIPLES": "label"})
    tokenizer = AutoTokenizer.from_pretrained(modelo_path)
    dataset = NeoplasiaDataset(df, tokenizer, max_len)

    return df, dataset, tokenizer


def validacion_cruzada(df, dataset, tokenizer, modelo_path, output_dir,
                       n_splits,  max_len, batch_size,
                       learning_rate, weight_decay,
                       max_epochs):
    

    device = torch.device("cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_auc = 0
    best_model_path = ""

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), df["label"])):
        print(f"\n Fold {fold+1}/{n_splits}")
        fold_dir = os.path.join(output_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

        model = AutoModelForSequenceClassification.from_pretrained(modelo_path, num_labels=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_auc = 0
        best_val_loss = float("inf")
        epochs_no_improve = 0

        with open(os.path.join(fold_dir, "metricas.txt"), "w") as f:
            f.write("Época\tTrainLoss\tValLoss\tValAcc\tValF1\tValAUC\n")

        early_stopper = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)


        for epoch in range(max_epochs):
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            # Validación
            model.eval()
            val_loss = 0
            val_preds, val_probs, val_labels = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            val_auc = roc_auc_score(val_labels, val_probs)

            with open(os.path.join(fold_dir, "metricas.txt"), "a") as f:
                f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\t{val_auc:.4f}\n")

            # Early stopping y mejor modelo
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                model.save_pretrained(os.path.join(fold_dir, "mejor_modelo"))
                tokenizer.save_pretrained(os.path.join(fold_dir, "mejor_modelo"))
                if best_val_auc > best_auc:
                    best_auc = best_val_auc
                    best_model_path = os.path.join(fold_dir, "mejor_modelo")
            else:
                if early_stopper(val_auc):
                    print(f"Early stopping activado en epoch {epoch+1} (AUC={val_auc:.4f})")

                    break

    return best_model_path, best_auc

def evaluar_en_test(model_path, test_df, output_dir, max_len=512, batch_size=16):

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)



    test_dataset = NeoplasiaDataset(test_df, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    os.makedirs(output_dir, exist_ok=True)

    # Métricas finales
    report = classification_report(test_labels, test_preds, digits=4)
    cm = confusion_matrix(test_labels, test_preds)
    auc_score = roc_auc_score(test_labels, test_probs)
    f1 = f1_score(test_labels, test_preds)

    sens = recall_score(test_labels, test_preds, pos_label=1)
    esp = recall_score(test_labels, test_preds, pos_label=0)

    mcc = matthews_corrcoef(test_labels, test_preds)


    # Guardar métricas
    with open(os.path.join(output_dir, "metricas_test.txt"), "w") as f:
        f.write("Reporte de clasificación:\n")
        f.write(report + "\n")
        f.write(f"AUC: {auc_score:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Sensibilidad: {sens:.4f}\n")
        f.write(f"Especifidad: {esp:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")

    # Matriz de confusión
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión - Test")
    plt.colorbar()
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "matriz_confusion_test.png"))
    plt.close()

    

    print("Evaluación en test completada. Resultados guardados en:", output_dir)




## ────────────────────────────────────────────────────────────── 
# MAIN
## ──────────────────────────────────────────────────────────────


# Rutas
output_dir = "../output/roberta/validación_cruzada/cv2"
os.makedirs(output_dir, exist_ok=True)
ruta_csv = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/Multiples_neoplasias_solo_resumenes_selection/textos_cortos_filtrados.csv"

# Parámetros
PATIENCE = 3
MIN_DELTA = 0.001
modelo_path = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
batch_size = 16
epochs = 10
max_len = 512
n_splits = 5
learning_rate=1e-5
weight_decay=0.01


# Cargar datos
df_total, dataset, tokenizer = cargarDatos(ruta_csv, modelo_path, max_len)

# Separar test
df_trainval, df_test = train_test_split(df_total, test_size=0.2, stratify=df_total["label"], random_state=42)
dataset_trainval = NeoplasiaDataset(df_trainval, tokenizer, max_len)

# Validación cruzada
best_model_path, best_auc = validacion_cruzada(df_trainval, dataset_trainval, tokenizer,
                                               modelo_path, output_dir,
                                               n_splits=n_splits,
                                               max_len=max_len,
                                               batch_size=batch_size,
                                               learning_rate=learning_rate,
                                               weight_decay=weight_decay,
                                               max_epochs=epochs
                                               )

# Evaluar en test
evaluar_en_test(best_model_path, df_test, os.path.join(output_dir, "test_final"), max_len=max_len, batch_size=batch_size)







