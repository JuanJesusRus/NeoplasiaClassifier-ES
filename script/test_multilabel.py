import ast
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Dataset
# -------------------------

import unicodedata

from train_multilabel import (
    normalize_token_to_canonical,
    reverse_syn_map
)



def normalize_label(label: str) -> str:
    """
    Normalización ligera para mapeo de etiquetas:
    - minúsculas
    - strip de espacios
    - eliminación de tildes
    """
    label = label.strip().lower()
    label = unicodedata.normalize("NFD", label)
    label = "".join(
        c for c in label
        if unicodedata.category(c) != "Mn"
    )
    return label



class MultilabelDataset(Dataset):
    def __init__(self, df, tokenizer, labels_list, max_length=512):
        self.texts = df["TEXTO"].tolist()
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label2idx = {
            normalize_label(l): i
            for i, l in enumerate(labels_list)
        }

        canonical_set = set(labels_list)

        self.targets = []

        for labs in df["NEOPLASIAS"]:
            vec = np.zeros(len(labels_list), dtype=int)

            for p in labs:
                canon = normalize_token_to_canonical(
                    p,
                    reverse_syn_map,
                    canonical_set,
                    case_sensitive=False   # MISMO valor que en train
                )
                if canon is not None:
                    vec[self.label2idx[normalize_label(canon)]] = 1

            self.targets.append(vec)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item



# -------------------------
# Evaluación
# -------------------------
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.append(labels)
            y_prob.append(probs)

    y_true = np.vstack(y_true)
    y_prob = np.vstack(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # Métricas globales
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    
    # AUC por etiqueta (solo si es válido)
    valid_aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc_i = roc_auc_score(y_true[:, i], y_prob[:, i])
                valid_aucs.append(auc_i)
            except ValueError:
                pass

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    # Micro-AUC solo si es válido
    try:
        if len(np.unique(y_true.ravel())) > 1:
            micro_auc = roc_auc_score(y_true.ravel(), y_prob.ravel())
        else:
            micro_auc = None
    except ValueError:
        micro_auc = None


    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc
    }


# -------------------------
# Confusion matrices
# -------------------------
def save_confusion_matrices(y_true, y_pred, labels_list, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, label in enumerate(labels_list):
        cm = confusion_matrix(
            y_true[:, i],
            y_pred[:, i],
            labels=[0, 1]
        )

        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(label)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out_dir / f"{label}.png")
        plt.close()


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    # Cargar datos
    df = pd.read_csv(args.test_csv, sep=";")
    


    df["NEOPLASIAS"] = df["NEOPLASIAS"].apply(ast.literal_eval)


    with open("C:\\Users\\jesus\\OneDrive - Universidad de Málaga\\Cuarto\\TFG\\NeoplasiaClassifier-ES\\output\\multilabel_run\\label2idx.json", "r", encoding="latin-1") as f:
        label2idx = json.load(f)


    labels_list = [None] * len(label2idx)
    for label, idx in label2idx.items():
        labels_list[idx] = label




    dataset = MultilabelDataset(df, tokenizer, labels_list)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate(model, loader, device, args.threshold)

    # Guardar métricas
    with open(out_dir / "test_metrics.json", "w", encoding="utf8") as f:
        json.dump(
            {
                "macro_auc": metrics["macro_auc"],
                "micro_auc": metrics["micro_auc"],
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
            },
            f,
            indent=4
        )

    # Guardar matrices
    save_confusion_matrices(
        metrics["y_true"],
        metrics["y_pred"],
        labels_list,
        out_dir / "confusion_matrices"
    )

    print("\n=== TEST RESULTS ===")
    print(f"Macro-AUC: {metrics['macro_auc'] if metrics['macro_auc'] is not None else 'N/A'}")
    print(f"Micro-AUC: {metrics['micro_auc'] if metrics['micro_auc'] is not None else 'N/A'}")

    print(f"Macro-F1 : {metrics['macro_f1']:.4f}")
    print(f"Micro-F1 : {metrics['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
