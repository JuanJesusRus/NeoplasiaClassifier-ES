# ============================================================
# neoplasia_train.py  
# ============================================================
import random
import argparse
import logging
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import (StratifiedKFold, train_test_split)
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
import matplotlib.pyplot as plt


def set_global_seed(seed: int):
    logging.info("Semilla global: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


# ------------------------------- logging -------------------------------
def setup_logging(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = Path(out_dir) / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ],
        force = True
    )
    logging.info("Logger inicializado en %s", log_path)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None

    def step(self, metric: float) -> bool:
        if self.best_metric is None or metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class NeoplasiaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        texto = str(self.df.iloc[idx]["texto"])
        label = int(self.df.iloc[idx]["label"])
        toks = self.tokenizer(
            texto,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def prepare_data(cfg) :
    
    df = (pd.read_csv(cfg["datos"]["ruta_csv"])
            .rename(columns={
                cfg["datos"]["columna_texto"]: "texto",
                cfg["datos"]["columna_etiqueta"]: "label"}))

    tokenizer = AutoTokenizer.from_pretrained(cfg["modelo"]["path"])
    max_len = cfg["modelo"].get("max_len", 512)

    df_trainval, df_test = train_test_split(
        df,
        test_size=float(cfg["datos"]["test_size"]),
        stratify=df["label"],
        random_state=int(cfg["datos"]["random_state"])
    )
    dataset_trainval = NeoplasiaDataset(df_trainval, tokenizer, max_len)
    return df_trainval, df_test, dataset_trainval, tokenizer


# -------------------------- evaluación test ----------------------------
def evaluar_en_test(model_path: str,
                    test_df: pd.DataFrame,
                    out_dir: Path,
                    tokenizer_path: str,
                    max_len: int,
                    batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    test_loader = DataLoader(
        NeoplasiaDataset(test_df, tokenizer, max_len),
        batch_size=batch_size)

    model.eval()
    preds, labels, probs = [], [], []
    idx_global = 0
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            ).logits
            probs_batch = torch.softmax(logits, 1)[:, 1]
            preds_batch = torch.argmax(logits, 1)
            labels_batch = batch["label"].numpy()

            # ---------- logging de errores ----------
            for p, y, prob in zip(preds_batch, labels_batch, probs_batch):
                if p != y:
                    # texto original correspondiente a este índice global
                    texto_orig = test_df.iloc[idx_global]["texto"]
                    logging.warning(
                        "Misclasificado (idx=%d) – pred=%d, real=%d, prob=%.3f | texto=\"%s\"",
                        idx_global, int(p), int(y), float(prob), texto_orig.replace("\n", " ")[:300]
                    )
                idx_global += 1
            # ----------------------------------------



            preds.extend(preds_batch.cpu().numpy())
            probs.extend(probs_batch.cpu().numpy())
            labels.extend(batch["label"].numpy())

    out_dir.mkdir(parents=True, exist_ok=True)
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    (out_dir / "metricas_test.txt").write_text(
        f"AUC: {auc:.4f}\n"
        f"Accuracy: {acc:.4f}\n"
        f"F1: {f1:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall: {rec:.4f}\n"
        f"\nMatriz de Confusión:\n{cm}\n"
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión - TEST")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Una neoplasia", "Múltiples"], rotation=45)
    plt.yticks(tick_marks, ["Una neoplasia", "Múltiples"])

    # Plotear Confussion Matrix
    # Añadir los números dentro de cada celda
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/matriz_confusion.png")
    plt.close()



    logging.info("Test – AUC %.4f | F1 %.4f | Accuracy %.4f", auc, f1, acc)


# -------------------- entrenamiento validación cruzada -----------------
def train_fold(fold_id: int,
               train_idx: List[int],
               val_idx: List[int],
               dataset: NeoplasiaDataset,
               tokenizer,
               cfg,
               device,
               out_dir: Path) -> Tuple[str, float]:
    logging.info("· Fold %d", fold_id)
    fold_dir = out_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    bs = int(cfg["entrenamiento"]["batch_size"])
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=bs)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["modelo"]["path"], num_labels=2).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["entrenamiento"]["learning_rate"]),
        weight_decay=float(cfg["entrenamiento"]["weight_decay"]))
    stopper = EarlyStopping(int(cfg["entrenamiento"]["patience"]),
                            float(cfg["entrenamiento"]["min_delta"]))

    best_auc, best_model_path = 0., ""
    for epoch in range(int(cfg["entrenamiento"]["num_epochs"])):
        model.train()
        for batch in tqdm(train_loader,
                          desc=f"Fold {fold_id} – Epoch {epoch+1}"):
            optim.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device))
            out.loss.backward()
            optim.step()

        # validación
        model.eval(); logits_lst, labels_lst = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)).logits
                logits_lst.append(logits.cpu())
                labels_lst.append(batch["label"].cpu())
        logits = torch.cat(logits_lst)
        labels = torch.cat(labels_lst)
        probs = torch.softmax(logits, 1)[:, 1]
        preds = torch.argmax(logits, 1)
        val_auc = roc_auc_score(labels, probs)
        val_f1 = f1_score(labels, preds)
        logging.info("    Epoch %d | AUC %.4f | F1 %.4f",
                     epoch+1, val_auc, val_f1)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_path = fold_dir / "best_model"
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)

        if stopper.step(val_auc):
            logging.info("    >> Early-stopping en epoch %d", epoch+1)
            break

    return str(best_model_path), best_auc


def run_cross_validation(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["salida"]["carpeta_resultados"])

    df_trainval, df_test, dataset, tokenizer = prepare_data(cfg)
    skf = StratifiedKFold(
        n_splits=int(cfg["entrenamiento"]["num_folds"]),
        shuffle=True, random_state=42)

    best_auc_global, best_model_global = 0., ""
    for fold_id, (tr_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(df_trainval)), df_trainval["label"]), start=1):
        model_path, auc = train_fold(
            fold_id, tr_idx, val_idx,
            dataset, tokenizer, cfg, device, out_dir)
        if auc > best_auc_global:
            best_auc_global, best_model_global = auc, model_path

    logging.info("Mejor modelo global AUC: %.4f (%s)",
                 best_auc_global, best_model_global)

    evaluar_en_test(best_model_global,
                    df_test,
                    out_dir / "test_final",
                    best_model_global,  # tokenizer en la misma carpeta
                    int(cfg["modelo"]["max_len"]),
                    int(cfg["entrenamiento"]["batch_size"]))


# ---------------------- entrenamiento simple ---------------------------
def run_simple(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["salida"]["carpeta_resultados"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # datos
    df = (pd.read_csv(cfg["datos"]["ruta_csv"])
            .rename(columns={
                cfg["datos"]["columna_texto"]: "texto",
                cfg["datos"]["columna_etiqueta"]: "label"}))
    df_train, df_temp = train_test_split(
        df, test_size=float(cfg["datos"]["test_size"]),
        stratify=df["label"],
        random_state=int(cfg["datos"]["random_state"]))
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5,
        stratify=df_temp["label"],
        random_state=int(cfg["datos"]["random_state"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg["modelo"]["path"])
    max_len = int(cfg["modelo"]["max_len"])
    bs = int(cfg["entrenamiento"]["batch_size"])

    train_loader = DataLoader(
        NeoplasiaDataset(df_train, tokenizer, max_len), batch_size=bs, shuffle=True)
    val_loader = DataLoader(
        NeoplasiaDataset(df_val, tokenizer, max_len), batch_size=bs)
    test_loader = DataLoader(
        NeoplasiaDataset(df_test, tokenizer, max_len), batch_size=bs)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["modelo"]["path"], num_labels=2).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["entrenamiento"]["learning_rate"]),
        weight_decay=float(cfg["entrenamiento"]["weight_decay"]))
    stopper = EarlyStopping(int(cfg["entrenamiento"]["patience"]),
                            float(cfg["entrenamiento"]["min_delta"]))

    best_auc = 0.
    for epoch in range(int(cfg["entrenamiento"]["num_epochs"])):
        # entrenamiento
        model.train(); total_loss = 0.
        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch+1}/{int(cfg['entrenamiento']['num_epochs'])}"):
            optim.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device))
            out.loss.backward(); optim.step()
            total_loss += out.loss.item()
        avg_loss = total_loss / len(train_loader)

        # validación
        model.eval(); preds, labs = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)).logits
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                labs.extend(batch["label"].numpy())
        val_auc = roc_auc_score(labs, preds)
        logging.info("Epoch %d | TrainLoss %.4f | ValAUC %.4f",
                     epoch+1, avg_loss, val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            model.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if stopper.step(val_auc):
                logging.info(">> Early-stopping en epoch %d", epoch+1)
                break

    # test
    evaluar_en_test(out_dir,
                    df_test,
                    out_dir / "test_final",
                    out_dir,  
                    max_len,
                    bs)


# ------------------------------- main ----------------------------------
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Neoplasia-ES unified script")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Ruta del archivo YAML de configuración")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["datos"]["random_state"]))
    setup_logging(cfg["salida"]["carpeta_resultados"])
    logging.info("Configuración cargada de %s", args.config)

    if cfg["entrenamiento"]["usar_validacion_cruzada"]:
        logging.info(">>> Modo VALIDACIÓN CRUZADA")
        run_cross_validation(cfg)
    else:
        logging.info(">>> Modo entrenamiento simple")
        run_simple(cfg)

    logging.info("Trabajo completado con éxito ")


if __name__ == "__main__":
    main()
