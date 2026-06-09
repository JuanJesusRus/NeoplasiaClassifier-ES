"""Entrenamiento y evaluación multilabel (único script)

Uso típico:
  python script/train_multilabel.py \
    --csv output/comparacionModelos/datos/train_set_cambiado.csv \
    --model-base PlanTL-GOB-ES/bsc-bio-ehr-es \
    --output-dir output/multilabel_run --epochs 5

Características:
- Detecta automáticamente el formato de etiquetas:
  * Formato B (por defecto): columna `LABELS` con etiquetas separadas por `;`
  * Formato A: columnas binarias por etiqueta (0/1)
- Construye multi-hot targets, entrena con BCEWithLogitsLoss y guarda modelo/tokenizer
- Evalúa usando macro-F1 (criterio para guardar mejor modelo) y métricas por etiqueta
- Modo `--predict` para generar predicciones y probabilidades desde un CSV o desde stdin
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple
import ast
import re
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

reverse_syn_map = {}


def setup_logging(output_dir: str):
    """Configura logging para guardar en run.log y consola."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = Path(output_dir) / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info("Logger inicializado en %s", log_path)


class EarlyStopping:
    """Early stopping para evitar overfitting.
    
    Monitorea una métrica (ej: macro_f1) y para el entrenamiento si no mejora
    durante `patience` epochs consecutivos.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        """
        Args:
            patience: número de epochs sin mejora antes de parar
            min_delta: cambio mínimo para considerar como mejora
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None

    def step(self, metric: float) -> bool:
        """
        Actualiza early stopping con nueva métrica.
        
        Args:
            metric: valor actual de la métrica a monitorear
        
        Returns:
            True si se debe detener el entrenamiento, False en caso contrario
        """
        if self.best_metric is None or metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def parse_collections_file(path: str):
    """Parsea un fichero tipo `colecciones = { ... }` y devuelve (labels_list, synonyms_map).
    - labels_list: lista de claves en el fichero (orden conservado)
    - synonyms_map: dict[label] -> list(synonyms...]
    """
    text = Path(path).read_text(encoding='utf-8')
    m = re.search(r"colecciones\s*=\s*\{", text)
    if not m:
        raise ValueError(f"No se encontró 'colecciones = {{...}}' en {path}")
    start = m.start()
    brace_idx = text.find('{', start)
    if brace_idx == -1:
        raise ValueError("Diccionario no encontrado.")
    depth = 0
    end_idx = None
    for i in range(brace_idx, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx is None:
        raise ValueError("No se pudo localizar el cierre del diccionario en el archivo de colecciones.")

    dict_text = text[brace_idx:end_idx+1]
    try:
        parsed = ast.literal_eval(dict_text)
    except Exception as e:
        raise ValueError(f"Error al evaluar el dict de colecciones: {e}")

    labels_list = list(parsed.keys())
    synonyms_map = {k: [s for s in v] for k, v in parsed.items()}
    return labels_list, synonyms_map


def build_labels_from_B(df: pd.DataFrame, labels_col: str = "LABELS", sep: str = ";") -> List[str]:
    labels = set()
    for raw in df[labels_col].dropna():
        s = str(raw).strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                parts = ast.literal_eval(s)
            except Exception:
                parts = [p.strip() for p in s.strip("[]") .split(sep) if p.strip()]
        else:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
        labels.update(parts)
    labels_list = sorted(labels)
    return labels_list


def normalize_token_variations(token: str) -> str:
    """
    Normaliza variaciones comunes de un token sin modificar la coleccion.
    Maneja plurales, acentos y variaciones ortográficas.
    
    Ejemplos:
    - "Linfomas no Hodgkin" → "Linfoma no hodgkin"
    - "Linfomas Hodgkin" → "Linfoma hodgkin"
    - "Intestino delgado" → "Intestino Delgado"
    - "Via biliar" → "Vía biliar"
    """
    t = token.strip()
    if not t:
        return t
    
    replacements = {
        "linfomas no hodgkin": "linfoma no hodgkin",
        "linfomas hodgkin": "linfoma hodgkin",
        "intestino delgado": "Intestino Delgado",
        "via biliar": "Vía biliar",
        "metástasis de origen desconocido": "Metástasis de origen desconocido",
        "canal anal y ano": "Recto",
        "GIST": "Sarcoma",
        "vesícula biliar": "Vía biliar",
        "suprarrenal": "Suprarrenal",
    }
    
    t_lower = t.lower()
    if t_lower in replacements:
        return replacements[t_lower]
    
    if t_lower.endswith("s") and len(t) > 3:
        singular = t[:-1]
        return singular
    
    return t


def normalize_token_to_canonical(token: str, reverse_syn_map: dict, canonical_set: set, case_sensitive: bool = True):
    """Mapea un token a la etiqueta canónica usando el mapa de sinónimos.

    Si `case_sensitive` es True, la comparación usa coincidencias exactas; si es False,
    se realiza matching case-insensitive (buscando lowercase en el reverse_syn_map).
    
    Primero normaliza variaciones comunes (plurales, acentos) antes de buscar en sinónimos.
    """
    t = token.strip()
    if not t:
        return None
    
    t = normalize_token_variations(t)
    
    if t in canonical_set:
        return t

    if not case_sensitive:
        for c in canonical_set:
            if t.lower() == c.lower():
                return c
        key = t.lower()
    else:
        key = t

    if key in reverse_syn_map:
        return reverse_syn_map[key]
    return None


from collections import Counter

def simulate_case_comparison(df: pd.DataFrame, labels_col: str, labels_list: List[str], synonyms_map: dict, sep: str = ';', top_n: int = 20, output_dir: str = None, no_normalize: bool = False):
    """Realiza una comprobación no destructiva comparando mapping case-sensitive vs case-insensitive.

    Imprime resumen de filas totales, filas sin etiquetas, filas con todas etiquetas no reconocidas,
    top etiquetas canónicas encontradas y top tokens no reconocidos para ambas estrategias.
    
    Si output_dir es proporcionado, guarda los resultados en sanity_check_results.txt
    Si no_normalize=True, desactiva las normalizaciones para mostrar resultados sin ellas.
    """
    cs_reverse = {}
    ci_reverse = {}
    for k, vals in synonyms_map.items():
        cs_reverse[k] = k
        ci_reverse[k.lower()] = k
        if not no_normalize:
            for v in vals:
                cs_reverse[v] = k
                ci_reverse[v.lower()] = k

    def compute(reverse_map, case_insensitive=False):
        canon_counts = Counter()
        unmatched = Counter()
        rows_with_no_labels = 0
        rows_all_unmatched = 0
        for _, row in df.iterrows():
            raw = row.get(labels_col, '')
            if pd.isna(raw) or str(raw).strip() == '':
                rows_with_no_labels += 1
                continue
            s = str(raw).strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    parts = ast.literal_eval(s)
                except Exception:
                    parts = [p.strip() for p in s.strip('[]').split(sep) if p.strip()]
            else:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
            matched_any = False
            for p in parts:
                key = p if not case_insensitive else p.lower()
                if key in reverse_map:
                    canon_counts[reverse_map[key]] += 1
                    matched_any = True
                    continue
                if not case_insensitive:
                    if p in labels_list:
                        canon_counts[p] += 1
                        matched_any = True
                        continue
                else:
                    for c in labels_list:
                        if p.lower() == c.lower():
                            canon_counts[c] += 1
                            matched_any = True
                            break
                    if matched_any:
                        continue
                unmatched[p] += 1
            if not matched_any:
                rows_all_unmatched += 1
        return canon_counts, unmatched, rows_with_no_labels, rows_all_unmatched

    cs_counts, cs_unmatched, cs_no_labels, cs_all_unmatched = compute(cs_reverse, case_insensitive=False)
    ci_counts, ci_unmatched, ci_no_labels, ci_all_unmatched = compute(ci_reverse, case_insensitive=True)

    total_rows = len(df)
    
    output_lines = []
    output_lines.append("\n--- Sanity check: case-sensitive vs case-insensitive mapping ---")
    output_lines.append(f"Total filas: {total_rows}")
    output_lines.append("Case-sensitive: rows_all_unmatched={}, rows_without_labels={}".format(cs_all_unmatched, cs_no_labels))
    output_lines.append("Case-insensitive: rows_all_unmatched={}, rows_without_labels={}".format(ci_all_unmatched, ci_no_labels))
    output_lines.append('\nTop canonical tags (case-sensitive):')
    for lab, cnt in cs_counts.most_common(top_n):
        output_lines.append(f"{lab}: {cnt}")
    output_lines.append('\nTop canonical tags (case-insensitive):')
    for lab, cnt in ci_counts.most_common(top_n):
        output_lines.append(f"{lab}: {cnt}")

    output_lines.append('\nTop unmatched tokens (case-sensitive):')
    for tok, cnt in cs_unmatched.most_common(top_n):
        output_lines.append(f"{tok}: {cnt}")
    output_lines.append('\nTop unmatched tokens (case-insensitive):')
    for tok, cnt in ci_unmatched.most_common(top_n):
        output_lines.append(f"{tok}: {cnt}")

    output_lines.append('--- End sanity check ---\n')
    
    for line in output_lines:
        print(line)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = Path(output_dir) / "sanity_check_results.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"[OK] Resultados guardados en: {result_file}")


def detect_format(df: pd.DataFrame, text_col: str = "TEXTO") -> Tuple[str, List[str]]:
    """Detecta formato A (columnas binarias) o B (columna LABELS).
    Retorna (format, labels_columns_or_name).
    """
    if "LABELS" in df.columns:
        return "B", ["LABELS"]

    label_cols = [c for c in df.columns if c != text_col]
    if not label_cols:
        raise ValueError("No se encontraron columnas de etiquetas ni columna 'LABELS'.")

    binary_ok = True
    for c in label_cols:
        vals = df[c].dropna().unique()
        allowed = set([0, 1])
        try:
            vals_conv = set(int(x) for x in vals)
        except Exception:
            binary_ok = False
            break
        if not vals_conv.issubset(allowed):
            binary_ok = False
            break
    if binary_ok:
        return "A", label_cols

    raise ValueError("Formato de etiquetas ambiguo. Pase --format A|B o asegúrese de que las columnas de etiquetas sean binarias o exista columna 'LABELS'.")


def filter_labels_exclude_mama(df: pd.DataFrame, labels_col: str = "LABELS", sep: str = ";", enable: bool = False) -> pd.DataFrame:
    """Filtra filas según reglas específicas para el modelo de RESTO.
    
    Solo aplica si enable=True. Las reglas son:
    -  Eliminar filas con SOLO ['Mama'] o ['Mama','Mama']
    -  Si la fila contiene ['Mama','Pulmón'], cambia a ['Pulmón']
    -  Mantiene ['Sarcoma'], ['Colon','Recto'] y otras combinaciones
    
    Args:
        df: DataFrame con la columna de etiquetas
        labels_col: nombre de la columna de etiquetas
        sep: separador de etiquetas
        enable: si False, retorna df sin cambios; si True, aplica el filtrado
    
    Returns:
        DataFrame filtrado y/o modificado
    """
    if not enable:
        return df.copy()
    
    logging.info("=" * 70)
    logging.info("Aplicando filtrado de Mama para modelo RESTO")
    logging.info("=" * 70)
    
    df_filtered = df.copy()
    rows_to_drop = []
    
    for idx, row in df_filtered.iterrows():
        raw = row.get(labels_col, '')
        if pd.isna(raw) or str(raw).strip() == '':
            continue
        
        s = str(raw).strip()
        
        if s.startswith('[') and s.endswith(']'):
            try:
                labels_list = ast.literal_eval(s)
                if isinstance(labels_list, str):
                    labels_list = [p.strip() for p in labels_list.split(sep) if p.strip()]
                else:
                    labels_list = [str(l).strip() for l in labels_list]
            except Exception:
                labels_list = [p.strip() for p in s.strip("[]").split(sep) if p.strip()]
        else:
            labels_list = [p.strip() for p in s.split(sep) if p.strip()]
        
        if labels_list == ['Mama'] or labels_list == ['Mama', 'Mama']:
            rows_to_drop.append(idx)
            continue
        
        if 'Mama' in labels_list and len(labels_list) > 1:
            new_labels = [l for l in labels_list if l != 'Mama']
            if new_labels:  # Si quedan etiquetas después de remover Mama
                # Conservar el formato original: si la representación original parecía
                # una lista Python (comenzaba y terminaba con corchetes), mantener
                # ese formato usando repr(); en caso contrario mantener el formato
                # separado por `sep` (p. ej. 'Pulmón;Colon').
                if s.startswith('[') and s.endswith(']'):
                    df_filtered.at[idx, labels_col] = repr(new_labels)
                else:
                    df_filtered.at[idx, labels_col] = sep.join(new_labels)
    
    # Aplicar drop
    df_filtered = df_filtered.drop(rows_to_drop, axis=0).reset_index(drop=True)
    
    logging.info(f"Filas eliminadas (solo Mama): {len(rows_to_drop)}")
    logging.info(f"Filas totales después de filtrado: {len(df_filtered)} (de {len(df)})")
    
    return df_filtered


class MultilabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, labels_list: List[str], max_len: int = 512, text_col: str = "TEXTO", format_type: str = "B", sep: str = ";", labels_col: str = "LABELS", unknown_policy: str = "other", labels_case_sensitive: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.labels_list = labels_list
        self.max_len = max_len
        self.text_col = text_col
        self.format_type = format_type
        self.sep = sep
        self.labels_col = labels_col
        self.unknown_policy = unknown_policy
        self.labels_case_sensitive = labels_case_sensitive

    def __len__(self):
        return len(self.df)

    def _get_label_vector(self, idx):
        if self.format_type == "A":
            row = self.df.iloc[idx][self.labels_list].fillna(0).astype(int).values
            return torch.tensor(row, dtype=torch.float)
        else:  # B
            raw = self.df.iloc[idx][self.labels_col]
            if pd.isna(raw) or str(raw).strip() == "":
                return torch.zeros(len(self.labels_list), dtype=torch.float)

            s = str(raw).strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    parts = ast.literal_eval(s)
                except Exception:
                    parts = [p.strip() for p in s.strip("[]") .split(self.sep) if p.strip()]
            else:
                parts = [p.strip() for p in s.split(self.sep) if p.strip()]

            vec = np.zeros(len(self.labels_list), dtype=float)
            for p in parts:
                canon = normalize_token_to_canonical(p, reverse_syn_map, set(self.labels_list), case_sensitive=self.labels_case_sensitive)
                if canon:
                    vec[label2idx[canon]] = 1.0
                else:
                    if self.unknown_policy == 'other' and 'Otro' in label2idx:
                        vec[label2idx['Otro']] = 1.0
            return torch.tensor(vec, dtype=torch.float)

    def __getitem__(self, idx):
        texto = str(self.df.iloc[idx][self.text_col])
        tokens = self.tokenizer(texto, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = self._get_label_vector(idx)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }




def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, device, threshold=0.5):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu()
            probs = torch.sigmoid(logits).numpy()

            all_labels.append(labels)
            all_probs.append(probs)

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    per_label = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            try:
                aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
            except Exception:
                aucs.append(None)
        else:
            aucs.append(None)
    
    
    valid_aucs = [a for a in aucs if a is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0


    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "per_label": per_label,  
        "macro_f1": macro_f1,
        "aucs": aucs,
        "macro_auc": macro_auc
    }


def save_label_map(labels_list, out_dir: Path):
    out = out_dir / "label2idx.json"
    out.write_text(json.dumps({l: i for i, l in enumerate(labels_list)}, ensure_ascii=False, indent=2))


def save_confusion_matrices(y_true, y_pred, labels_list, out_dir: Path):
    """Guarda matrices de confusión por etiqueta (one-vs-rest) como imágenes y TXT.
    
    Para cada etiqueta, calcula una matriz 2x2:
    - True Negatives: (pred=0, true=0)
    - False Positives: (pred=1, true=0)
    - False Negatives: (pred=0, true=1)
    - True Positives: (pred=1, true=1)
    
    También guarda una matriz de confusión multilabel global mostrando las confusiones
    entre etiquetas (filas=reales, columnas=predichas).
    """
    out_dir = Path(out_dir)
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True, parents=True)
    
    cm_text = "=== Matrices de Confusión por Etiqueta (One-vs-Rest) ===\n\n"
    
    # Calcular matrices individuales
    individual_cms = {}
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    
    for label_idx, label_name in enumerate(labels_list):
        y_true_label = y_true[:, label_idx]
        y_pred_label = y_pred[:, label_idx]
        
        cm = confusion_matrix(y_true_label, y_pred_label, labels=[0, 1])
        individual_cms[label_name] = cm
        
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp
        
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión - {label_name}")
        plt.colorbar()
        plt.ylabel("Verdadero")
        plt.xlabel("Predicho")
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No (0)", "Sí (1)"])
        plt.yticks(tick_marks, ["No (0)", "Sí (1)"])
        
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.tight_layout()
        img_path = cm_dir / f"cm_{label_idx:02d}_{label_name}.png"
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        cm_text += f"{label_idx}. {label_name}:\n"
        cm_text += f"  Matriz:\n    {cm[0, 0]:5d} {cm[0, 1]:5d}\n    {cm[1, 0]:5d} {cm[1, 1]:5d}\n"
        cm_text += f"  TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}\n\n"
    
    global_cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    
    cm_text += "\n" + "="*70 + "\n"
    cm_text += "=== MATRIZ DE CONFUSIÓN GLOBAL AGREGADA ===\n"
    cm_text += "(Suma de todas las etiquetas)\n\n"
    cm_text += f"              No (0)    Sí (1)\n"
    cm_text += f"Verdadero No: {total_tn:7d}  {total_fp:7d}  (TN={total_tn}, FP={total_fp})\n"
    cm_text += f"Verdadero Sí: {total_fn:7d}  {total_tp:7d}  (FN={total_fn}, TP={total_tp})\n\n"
    
    global_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
    
    cm_text += f"Accuracy Global: {global_accuracy:.4f}\n"
    cm_text += f"Precision Global: {global_precision:.4f}\n"
    cm_text += f"Recall Global: {global_recall:.4f}\n"
    cm_text += f"F1-Score Global: {global_f1:.4f}\n"
    
    plt.figure(figsize=(7, 6))
    plt.imshow(global_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión Global Agregada (Todas las Etiquetas)")
    plt.colorbar()
    plt.ylabel("Verdadero")
    plt.xlabel("Predicho")
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No (0)", "Sí (1)"])
    plt.yticks(tick_marks, ["No (0)", "Sí (1)"])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(global_cm[i, j]), ha="center", va="center", 
                    color="white" if global_cm[i, j] > global_cm.max() / 2 else "black", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    global_img_path = cm_dir / "cm_global_agregada.png"
    plt.savefig(global_img_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    multilabel_cm = np.zeros((len(labels_list), len(labels_list)), dtype=int)
    
    for sample_idx in range(len(y_true)):
        true_labels = np.where(y_true[sample_idx] == 1)[0]  # índices de etiquetas verdaderas
        pred_labels = np.where(y_pred[sample_idx] == 1)[0]  # índices de etiquetas predichas
        
        if len(true_labels) == 0:
            continue
        
        for true_idx in true_labels:
            if len(pred_labels) == 0:
                multilabel_cm[true_idx, true_idx] -= 1  # Marcar como error (negativo para destacar)
            else:
                for pred_idx in pred_labels:
                    multilabel_cm[true_idx, pred_idx] += 1
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(multilabel_cm, interpolation='nearest', cmap=plt.cm.YlOrRd)
    plt.colorbar(im, ax=ax, label="Frecuencia")
    
    ax.set_xticks(np.arange(len(labels_list)))
    ax.set_yticks(np.arange(len(labels_list)))
    ax.set_xticklabels(labels_list, rotation=45, ha='right')
    ax.set_yticklabels(labels_list)
    ax.set_ylabel("Etiquetas Reales")
    ax.set_xlabel("Etiquetas Predichas")
    ax.set_title("Matriz de Confusión Multilabel\n(Filas=Reales, Columnas=Predichas)")
    
    for i in range(len(labels_list)):
        for j in range(len(labels_list)):
            if multilabel_cm[i, j] > 0:
                ax.text(j, i, str(multilabel_cm[i, j]), ha="center", va="center", 
                       color="white" if multilabel_cm[i, j] > multilabel_cm.max() / 2 else "black", fontsize=8)
    
    plt.tight_layout()
    multilabel_img_path = cm_dir / "cm_multilabel_etiquetas.png"
    plt.savefig(multilabel_img_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    multilabel_df = pd.DataFrame(multilabel_cm, index=labels_list, columns=labels_list)
    multilabel_csv_path = cm_dir / "cm_multilabel_etiquetas.csv"
    multilabel_df.to_csv(multilabel_csv_path, encoding='utf-8')
    
    cm_text += "\n" + "="*70 + "\n"
    cm_text += "=== MATRIZ DE CONFUSIÓN MULTILABEL (ETIQUETAS REALES vs PREDICHAS) ===\n"
    cm_text += "(Filas=Etiquetas Reales, Columnas=Etiquetas Predichas)\n"
    cm_text += "(Muestra cuántas veces se confunde una etiqueta real con otra)\n\n"
    cm_text += multilabel_df.to_string()
    cm_text += "\n\nEjemplo de lectura:\n"
    cm_text += "- Si Mama (fila) y Colon (columna) = 15, significa que 15 muestras \n"
    cm_text += "  tenían Mama como etiqueta real pero el modelo también predijo Colon.\n"
    
    cm_text_path = out_dir / "confusion_matrices.txt"
    cm_text_path.write_text(cm_text, encoding='utf-8')
    logging.info("✓ Matrices de confusión guardadas en: %s/", cm_dir)
    logging.info("  - Individuales por etiqueta: cm_00_*.png, cm_01_*.png, ...")
    logging.info("  - Global agregada: cm_global_agregada.png")
    logging.info("  - Multilabel (etiquetas): cm_multilabel_etiquetas.png")
    logging.info("  - CSV multilabel: cm_multilabel_etiquetas.csv")
    logging.info("✓ Resumen en: %s", cm_text_path)



def load_yaml_config(path: str) -> dict:
    """Carga un archivo YAML y devuelve un diccionario plano de configuración.

    El YAML puede contener cualquier clave de los argumentos del script (p. ej. `csv`, `model_base`,
    `labels_file`, `threshold`, etc.). Las claves del YAML usan _underscore_ (p.ej. `model_base`).
    """
    try:
        import yaml
    except Exception:
        raise ImportError("PyYAML no está instalado. Instala con 'pip install pyyaml' para usar archivos de config YAML.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Archivo de configuración YAML no encontrado: {path}")
    with p.open("r", encoding="utf-8") as f:
        conf = yaml.safe_load(f) or {}
    return conf


def main():
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", type=str, default=None, help="Archivo YAML con config (valores por defecto)")
    cfg_parser.add_argument("--output-dir", type=str, default=None, help="Directorio de salida (para setup logging)")
    cfg_known, _ = cfg_parser.parse_known_args()

    config = {}
    if cfg_known.config:
        config = load_yaml_config(cfg_known.config)
    else:
        default_cfg = Path(__file__).parent / "config_multilabel.yaml"
        if default_cfg.exists():
            config = load_yaml_config(str(default_cfg))
    
    output_dir = cfg_known.output_dir if cfg_known.output_dir else config.get("output_dir", "output/multilabel_run")
    setup_logging(output_dir)
    
    logging.info("="*70)
    logging.info("Iniciando entrenamiento multilabel")
    logging.info("="*70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=cfg_known.config, help="Archivo YAML con config (valores por defecto)")
    parser.add_argument("--csv", type=str, default=config.get("csv", "output/comparacionModelos/datos/train_set_completo.csv"))
    parser.add_argument("--val-csv", type=str, default=config.get("val_csv", None), help="CSV de validación (opcional; si no se proporciona, se divide del train set)")
    parser.add_argument("--test-csv", type=str, default=config.get("test_csv", None), help="CSV de test (opcional; para predicción/evaluación)")
    parser.add_argument("--text-col", type=str, default=config.get("text_col", "TEXTO"))
    parser.add_argument("--labels-col", type=str, default=config.get("labels_col", "NEOPLASIAS"), help="Nombre de la columna que contiene las etiquetas multilabel (por ejemplo 'NEOPLASIAS')")
    parser.add_argument("--labels-file", type=str, default=config.get("labels_file", "output/coleccion/coleccion_2.txt"), help="Archivo con el dict 'colecciones = { ... }' para etiquetas canónicas y sinónimos")
    parser.add_argument("--format", type=str, choices=["auto", "A", "B"], default=config.get("format", "auto"))
    parser.add_argument("--sep", type=str, default=config.get("sep", ";"))
    parser.add_argument("--model-base", type=str, default=config.get("model_base", "PlanTL-GOB-ES/bsc-bio-ehr-es"))
    parser.add_argument("--max-len", type=int, default=config.get("max_len", 512))
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", 16))
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 10))
    parser.add_argument("--lr", type=float, default=config.get("lr", 1e-5))
    parser.add_argument("--weight-decay",type=float, default=config.get("weight_decay", 0.01))
    parser.add_argument("--output-dir", type=str, default=config.get("output_dir", "output/multilabel_run"))
    parser.add_argument("--device", type=str, default=config.get("device", None))
    parser.add_argument("--threshold", type=float, default=config.get("threshold", 0.5))
    parser.add_argument("--val-size", type=float, default=config.get("val_size", 0.2))
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--min-support", type=int, default=config.get("min_support", 1), help="Eliminar etiquetas con soporte menor que este umbral")
    parser.add_argument("--unknown-policy", type=str, choices=["ignore", "other", "error"], default=config.get("unknown_policy", "other"), help="Qué hacer con etiquetas no listadas en --labels-file: ignore=omitir, other=mapear a 'Otro', error=lanzar excepción")
    parser.add_argument("--labels-case-sensitive", dest='labels_case_sensitive', action='store_true', help='Activar matching case-sensitive para etiquetas (default: True)')
    parser.add_argument("--no-labels-case-sensitive", dest='labels_case_sensitive', action='store_false', help='Desactivar matching case-sensitive (usa matching case-insensitive)')
    parser.set_defaults(labels_case_sensitive=config.get('labels_case_sensitive', True))
    parser.add_argument("--early-stopping-patience", type=int, default=config.get("early_stopping_patience", 3), help="Número de epochs sin mejora antes de parar (default: 3)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=config.get("early_stopping_min_delta", 0.001), help="Cambio mínimo para considerar como mejora (default: 0.001)")
    parser.add_argument("--early-stopping", action='store_true', default=config.get("early_stopping", False), help="Activar early stopping para prevenir overfitting")
    parser.add_argument("--sanity-check", action="store_true", help="Ejecuta una comprobación no destructiva comparando case-sensitive vs case-insensitive y sale")
    parser.add_argument("--no-normalize", action="store_true", help="En sanity-check, desactiva las normalizaciones para ver resultados sin ellas")
    parser.add_argument("--filter-exclude-mama", action='store_true', default=config.get("filter_exclude_mama", False), help="(Para modelo RESTO) Filtra filas con solo Mama y remueve Mama de combinaciones")
    parser.add_argument("--predict", action="store_true", help="Modo predicción: genera predicciones para el CSV dado y las guarda en output-dir/predictions.csv")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"⚠️ Aviso: el CSV '{csv_path}' no existe en el repositorio. Asegúrate de que la ruta es correcta.")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    df = pd.read_csv(csv_path, sep=args.sep, encoding='utf-8')
    df.columns = [c.strip() for c in df.columns]

    df = filter_labels_exclude_mama(df, labels_col=args.labels_col, sep=args.sep, enable=args.filter_exclude_mama)

    if args.format == "auto":
        if args.labels_col in df.columns:
            fmt = "B"
            label_cols = [args.labels_col]
        else:
            fmt, label_cols = detect_format(df, text_col=args.text_col)
    else:
        fmt = args.format
        if fmt == "B" and args.labels_col not in df.columns:
            raise ValueError(f"Formato B seleccionado pero no existe columna '{args.labels_col}' en CSV")
        if fmt == "A":
            label_cols = [c for c in df.columns if c != args.text_col]

    print(f"Formato detectado: {fmt}")

    global label2idx, reverse_syn_map

    labels_list = None
    synonyms_map = {}
    reverse_syn_map = {}

    labels_file_path = Path(args.labels_file)
    if labels_file_path.exists():
        labels_list, synonyms_map = parse_collections_file(str(labels_file_path))
        case_sensitive = args.labels_case_sensitive
        for canon, syns in synonyms_map.items():
            key = canon if case_sensitive else canon.lower()
            reverse_syn_map[key] = canon
            for s in syns:
                skey = s if case_sensitive else s.lower()
                reverse_syn_map[skey] = canon
    else:
        print(f"⚠️ Labels file {labels_file_path} no encontrado. Se derivarán etiquetas del CSV.")

    if labels_list is not None:
        labels_list = list(labels_list)
        if args.unknown_policy == 'other' and 'Otro' not in labels_list:
            labels_list.append('Otro')
    else:
        if fmt == 'B':
            labels_list = build_labels_from_B(df, labels_col=args.labels_col, sep=args.sep)
        else:
            labels_list = sorted(label_cols)

    if 'reverse_syn_map' not in globals():
        reverse_syn_map = globals().get('reverse_syn_map', {})

    if args.sanity_check:
        sim_syn_map = synonyms_map if synonyms_map else {}
        print("Running sanity check (non-destructive)...")
        if args.no_normalize:
            print("  (Sin normalizaciones aplicadas)")
        simulate_case_comparison(df, args.labels_col, labels_list, sim_syn_map, sep=args.sep, top_n=30, output_dir=args.output_dir, no_normalize=args.no_normalize)
        return

    support_counts = {l: 0 for l in labels_list}
    if fmt == 'B':
        for raw in df[args.labels_col].dropna():
            s = str(raw).strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    parts = ast.literal_eval(s)
                except Exception:
                    parts = [p.strip() for p in s.strip("[]") .split(args.sep) if p.strip()]
            else:
                parts = [p.strip() for p in s.split(args.sep) if p.strip()]
            parts_norm = set()
            for p in parts:
                canon = normalize_token_to_canonical(p, reverse_syn_map, set(labels_list), case_sensitive=args.labels_case_sensitive)
                if canon:
                    parts_norm.add(canon)
                else:
                    if args.unknown_policy == 'other':
                        parts_norm.add('Otro')
                    elif args.unknown_policy == 'error':
                        raise ValueError(f"Etiqueta desconocida encontrada en datos y policy=error: {p}")
            for c in parts_norm:
                if c in support_counts:
                    support_counts[c] += 1

    if args.min_support > 1:
        kept = [l for l in labels_list if support_counts.get(l, 0) >= args.min_support]
        removed = [l for l in labels_list if l not in kept]
        if removed:
            print(f"Etiquetas removidas por min_support ({args.min_support}): {removed}")
        labels_list = kept

    label2idx = {l: i for i, l in enumerate(labels_list)}

    if not labels_list:
        raise ValueError("No se detectaron etiquetas después de aplicar filtros.")

    print(f"Etiquetas ({len(labels_list)}): {labels_list}")
    save_label_map(labels_list, Path(args.output_dir))

    if synonyms_map:
        with open(Path(args.output_dir) / 'synonyms_map.json', 'w', encoding='utf-8') as f:
            json.dump(synonyms_map, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_base)

    if args.predict:
        model_dir = Path(args.output_dir) / "best_model"
        if not model_dir.exists():
            raise ValueError(f"No se encontró modelo en {model_dir}. Entrena primero o indica modelo.")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=len(labels_list)).to(device)
        model.eval()

        dataset = MultilabelDataset(df, tokenizer, labels_list, max_len=args.max_len, text_col=args.text_col, format_type=fmt, sep=args.sep, labels_col=args.labels_col, unknown_policy=args.unknown_policy, labels_case_sensitive=args.labels_case_sensitive)
        loader = DataLoader(dataset, batch_size=args.batch_size)

        out_rows = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predict"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits.cpu()).numpy()
                for p in probs:
                    labels_pred = [labels_list[i] for i, v in enumerate(p) if v >= args.threshold]
                    out_rows.append({"predicted": ";".join(labels_pred), **{f"prob_{labels_list[i]}": float(p[i]) for i in range(len(labels_list))}})

        out_df = pd.DataFrame(out_rows)

        if args.labels_col in df.columns:
            out_df.insert(1, "ETIQUETA_REAL", df[args.labels_col].values)
        out_df.to_csv(Path(args.output_dir) / "predictions.csv", index=False, encoding='utf-8')
        print("Predicciones guardadas en:", Path(args.output_dir) / "predictions.csv")
        return

    if args.val_csv:
        val_csv_path = Path(args.val_csv)
        if not val_csv_path.exists():
            raise FileNotFoundError(f"Archivo de validación no encontrado: {val_csv_path}")
        train_df = df
        val_df = pd.read_csv(val_csv_path, sep=args.sep, encoding='utf-8')
        val_df.columns = [c.strip() for c in val_df.columns]
        logging.info("Validación cargada desde: %s (%d filas)", args.val_csv, len(val_df))
    else:
        train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
        logging.info("Validación dividida del train set: %d filas (%.0f%%)", len(val_df), args.val_size*100)

    train_dataset = MultilabelDataset(train_df, tokenizer, labels_list, max_len=args.max_len, text_col=args.text_col, format_type=fmt, sep=args.sep, labels_col=args.labels_col, unknown_policy=args.unknown_policy, labels_case_sensitive=args.labels_case_sensitive)
    val_dataset = MultilabelDataset(val_df, tokenizer, labels_list, max_len=args.max_len, text_col=args.text_col, format_type=fmt, sep=args.sep, labels_col=args.labels_col, unknown_policy=args.unknown_policy, labels_case_sensitive=args.labels_case_sensitive)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_base, num_labels=len(labels_list)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    early_stop = None
    if args.early_stopping:
        early_stop = EarlyStopping(patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta)
        logging.info("Early stopping activado: patience=%d, min_delta=%.4f", args.early_stopping_patience, args.early_stopping_min_delta)
    
    logging.info("Iniciando entrenamiento - %d epochs, batch_size=%d, lr=%.2e", args.epochs, args.batch_size, args.lr)
    
    for epoch in range(1, args.epochs + 1):
        logging.info("\nEpoch %d/%d", epoch, args.epochs)
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        logging.info("  Train loss: %.4f", train_loss)

        metrics = eval_epoch(model, val_loader, device, threshold=args.threshold)
        logging.info("  Val macro-F1: %.4f", metrics['macro_f1'])

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": metrics['macro_f1'],
            "per_label_precision": metrics['per_label'][0].tolist(),
            "per_label_recall": metrics['per_label'][1].tolist(),
            "per_label_f1": metrics['per_label'][2].tolist(),
            "per_label_support": metrics['per_label'][3].tolist(),
            "aucs": metrics['aucs'],
            "val_macro_auc": metrics["macro_auc"],

        }
        with open(Path(args.output_dir) / f"epoch_{epoch}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(epoch_metrics, f, ensure_ascii=False, indent=2)

        if metrics['macro_auc'] > best_auc:
            best_auc = metrics['macro_auc']
            logging.info("  Nueva mejor macro-AUC: %.4f — guardando modelo...", best_auc)
            model.save_pretrained(Path(args.output_dir) / "best_model")
            tokenizer.save_pretrained(Path(args.output_dir) / "best_model")
        
        if early_stop:
            should_stop = early_stop.step(metrics['macro_auc'])
            if should_stop:
                logging.info("\n  Early stopping activado después de %d epochs sin mejora", early_stop.counter)
                logging.info("Mejor macro-AUC alcanzado: %.4f", early_stop.best_metric)
                break

    logging.info("Entrenamiento finalizado. Mejor macro-AUC: %.4f", best_auc)
    logging.info("Modelo y tokenizer guardados en: %s", Path(args.output_dir) / "best_model")
    
    if args.test_csv:
        test_csv_path = Path(args.test_csv)
        if test_csv_path.exists():
            logging.info("\n Evaluando en test set: %s", args.test_csv)
            test_df = pd.read_csv(test_csv_path, sep=args.sep, encoding='utf-8')
            test_df.columns = [c.strip() for c in test_df.columns]
            
            test_dataset = MultilabelDataset(test_df, tokenizer, labels_list, max_len=args.max_len, text_col=args.text_col, format_type=fmt, sep=args.sep, labels_col=args.labels_col, unknown_policy=args.unknown_policy, labels_case_sensitive=args.labels_case_sensitive)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
            
            test_metrics = eval_epoch(model, test_loader, device, threshold=args.threshold)
            
            model.eval()
            all_test_labels = []
            all_test_probs = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Test", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].cpu().numpy()
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.cpu()
                    probs = torch.sigmoid(logits).numpy()
                    
                    all_test_labels.append(labels)
                    all_test_probs.append(probs)
            
            y_test_true = np.vstack(all_test_labels)
            y_test_prob = np.vstack(all_test_probs)
            y_test_pred = (y_test_prob >= args.threshold).astype(int)
            
            save_confusion_matrices(y_test_true, y_test_pred, labels_list, Path(args.output_dir) / "test_results")
            
            test_metrics_txt = f"""=== Métricas de Test ===

Macro-F1: {test_metrics['macro_f1']:.4f}
Macro-AUC: {test_metrics['macro_auc']:.4f}

=== Métricas por Etiqueta ===
Label                         | Precision | Recall | F1-Score | Support
{'-'*75}
"""
            for i, label in enumerate(labels_list):
                prec = test_metrics['per_label'][0][i]
                rec = test_metrics['per_label'][1][i]
                f1 = test_metrics['per_label'][2][i]
                support = int(test_metrics['per_label'][3][i])
                test_metrics_txt += f"{label:<30} | {prec:>8.4f} | {rec:>6.4f} | {f1:>8.4f} | {support:>7}\n"
            
            test_metrics_txt += f"\n=== AUC por Etiqueta ===\n"
            for i, label in enumerate(labels_list):
                if label in test_metrics['aucs']:
                    auc = test_metrics['aucs'][label]
                    test_metrics_txt += f"{label}: {auc:.4f}\n"
            
            test_results_path = Path(args.output_dir) / "test_results" / "metricas_test.txt"
            test_results_path.parent.mkdir(parents=True, exist_ok=True)
            test_results_path.write_text(test_metrics_txt, encoding='utf-8')
            logging.info("✓ Métricas de test guardadas en: %s", test_results_path)
            logging.info("\nTest macro-F1: %.4f", test_metrics['macro_f1'])



if __name__ == "__main__":
    main()
