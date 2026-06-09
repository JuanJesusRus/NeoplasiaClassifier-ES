#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_cascade_tune.py

Script único para CASCADA con:
- mode=tune: seleccionar umbral(es) en VALIDACIÓN interna (sin tocar TEST)
- mode=test: evaluar en TEST usando el umbral seleccionado previamente

Además, en mode=test genera salidas "compatibles" con tu predict_cascade.py:
- resultados_cascada.csv (NEOPLASIAS, PREDICCION_CASCADA, PROBABILIDADES)
- resultados_cascada_metricas.json (macro_f1, macro_auc, métricas por etiqueta, etc.)
y también mantiene:
- test_metrics.json (panel moderno: micro/macro, exact_match, hamming, ruido)

Soporta config YAML (recomendado) + overrides por CLI.

Requisitos:
  pip install pyyaml transformers torch scikit-learn pandas numpy tqdm

Ejemplos:
  python script/predict_cascade_tune.py --config script/config_cascade_val.yaml
  python script/predict_cascade_tune.py --config script/config_cascade_test.yaml

YAML mínimo:
  mode: tune            # tune o test
  csv: output/.../val.csv
  sep: ";"
  text_col: "TEXTO"
  labels_col: "NEOPLASIAS"
  outdir: "output/multilabel_cascade_tuning"

  path_mama_model: "C:/.../multilabel_modelo_mama/best_model"
  path_mama_labels: "C:/.../multilabel_modelo_mama"
  path_resto_model: "C:/.../multilabel_modelo_resto/best_model"
  path_resto_labels: "C:/.../multilabel_modelo_resto"

  threshold_mama: 0.5
  thresholds_resto: [0.5, 0.4, 0.3, 0.25, 0.2]
  select_metric: "macro_f1"   # o "micro_recall"
  max_p90_labels: 3           # opcional: controla ruido
  best_threshold_json: "best_threshold.json"

Opcional (2 niveles):
  two_level: true
  support_cutoff: 10
  thresholds_major: [0.5, 0.4, 0.3]
  thresholds_minor: [0.3, 0.25, 0.2]

Notas:
- Ground truth se normaliza usando synonyms_map.json (si existe en path_resto_labels)
  y las funciones normalize_token_* de train_multilabel.py (mismo espíritu que tu predict_cascade.py).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    roc_auc_score,
    precision_recall_fscore_support,
)

# -------------------------------------------------------
# Normalización (igual espíritu que tu predict_cascade.py)
# -------------------------------------------------------
try:
    # Si tu repo mantiene el script dentro de script/, esto suele funcionar
    from script.train_multilabel import normalize_token_to_canonical, normalize_token_variations
except Exception:
    # Fallback: versiones simples (no ideal, pero evita romper ejecución)
    def normalize_token_variations(token: str) -> str:
        return str(token).strip()

    def normalize_token_to_canonical(token: str, reverse_syn_map: dict, canonical_set: set, case_sensitive: bool = True):
        t = str(token).strip()
        if not t:
            return None
        if not case_sensitive:
            # match case-insensitive con canónicas
            for c in canonical_set:
                if t.lower() == str(c).lower():
                    return c
            return reverse_syn_map.get(t.lower())
        return t if t in canonical_set else reverse_syn_map.get(t)


# -----------------------------
# YAML loader
# -----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el YAML: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


# -----------------------------
# Label maps
# -----------------------------
def load_label2idx(labels_dir: str) -> Dict[str, int]:
    p = Path(labels_dir) / "label2idx.json"
    if not p.exists():
        raise FileNotFoundError(f"No se encontró label2idx.json en: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except UnicodeDecodeError:
        # Igual que en tu predict_cascade.py: fallback a latin-1 (cp1252 en Windows)
        with p.open("r", encoding="latin-1") as f:
            d = json.load(f)

    return {str(k): int(v) for k, v in d.items()}


def invert_label2idx(label2idx: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in label2idx.items()}


def build_union_labelspace(
    mama_labels_dir: str,
    resto_labels_dir: str,
) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    mama_label2idx = load_label2idx(mama_labels_dir)
    resto_label2idx = load_label2idx(resto_labels_dir)

    mama_idx2label = invert_label2idx(mama_label2idx)
    resto_idx2label = invert_label2idx(resto_label2idx)

    mama_labels = [mama_idx2label[i] for i in sorted(mama_idx2label)]
    resto_labels = [resto_idx2label[i] for i in sorted(resto_idx2label)]

    all_labels = list(mama_labels)
    for lab in resto_labels:
        if lab not in all_labels:
            all_labels.append(lab)

    all_label2idx = {lab: i for i, lab in enumerate(all_labels)}
    return all_labels, all_label2idx, mama_label2idx, resto_label2idx, mama_idx2label, resto_idx2label


# -----------------------------
# Synonyms / reverse map loader
# -----------------------------
def load_synonyms_reverse_map(resto_labels_dir: str, all_labels: List[str]) -> Dict[str, str]:
    """
    Carga synonyms_map.json si existe en resto_labels_dir y construye reverse_syn_map (case-insensitive).
    Además, añade las etiquetas canónicas para que mapeen a sí mismas.
    """
    reverse_syn_map: Dict[str, str] = {}

    syn_path = Path(resto_labels_dir) / "synonyms_map.json"
    if syn_path.exists():
        try:
            syn_map = json.loads(syn_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            syn_map = json.loads(syn_path.read_text(encoding="latin-1"))

        for canon, syns in syn_map.items():
            reverse_syn_map[str(canon).lower()] = canon
            for s in syns:
                reverse_syn_map[str(s).lower()] = canon

    # asegurar que canónicas mapean a sí mismas
    for lab in all_labels:
        reverse_syn_map[str(lab).lower()] = lab

    return reverse_syn_map


# -----------------------------
# Parsing y_true
# -----------------------------
def parse_labels_cell(cell: Any, sep: str = ";") -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
        s = s.strip("[]")
    return [p.strip() for p in s.split(sep) if p.strip()]


def build_y_true(
    df: pd.DataFrame,
    labels_col: str,
    all_label2idx: Dict[str, int],
    sep: str,
    reverse_syn_map: Dict[str, str],
    canonical_set: set,
) -> np.ndarray:
    """
    Construye y_true multi-hot, NORMALIZANDO etiquetas reales como en tu pipeline:
    - normalize_token_variations()
    - normalize_token_to_canonical(..., case_sensitive=False)
    """
    y_true = np.zeros((len(df), len(all_label2idx)), dtype=int)
    for i, raw in enumerate(df[labels_col].values):
        labs = parse_labels_cell(raw, sep=sep)
        for lab in labs:
            t = normalize_token_variations(str(lab).strip())
            canon = normalize_token_to_canonical(t, reverse_syn_map, canonical_set, case_sensitive=False)
            if canon and canon in all_label2idx:
                y_true[i, all_label2idx[canon]] = 1
    return y_true


def compute_support(y_true: np.ndarray) -> np.ndarray:
    return y_true.sum(axis=0).astype(int)


# -----------------------------
# Inference
# -----------------------------
def load_model(model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_probs_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 16,
    max_len: int = 512,
) -> np.ndarray:
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def combine_probs(
    probs_mama_local: np.ndarray,
    probs_resto_local: np.ndarray,
    mama_idx2label: Dict[int, str],
    resto_idx2label: Dict[int, str],
    all_labels: List[str],
    all_label2idx: Dict[str, int],
) -> np.ndarray:
    """
    Combina probabilidades COMPLETAS (sin threshold).
    """
    n = probs_mama_local.shape[0]
    y_prob = np.zeros((n, len(all_labels)), dtype=float)

    # mama -> all
    for j in range(probs_mama_local.shape[1]):
        lab = mama_idx2label[j]
        y_prob[:, all_label2idx[lab]] = np.maximum(y_prob[:, all_label2idx[lab]], probs_mama_local[:, j])

    # resto -> all
    for j in range(probs_resto_local.shape[1]):
        lab = resto_idx2label[j]
        if lab in all_label2idx:
            y_prob[:, all_label2idx[lab]] = np.maximum(y_prob[:, all_label2idx[lab]], probs_resto_local[:, j])

    return y_prob


# -----------------------------
# Thresholding
# -----------------------------
def apply_thresholds_global(
    y_prob: np.ndarray,
    all_labels: List[str],
    t_mama: float,
    t_resto: float,
    mama_label_name: str = "Mama",
) -> np.ndarray:
    y_pred = (y_prob >= t_resto).astype(int)
    if mama_label_name in all_labels:
        j = all_labels.index(mama_label_name)
        y_pred[:, j] = (y_prob[:, j] >= t_mama).astype(int)
    return y_pred


def apply_thresholds_two_level(
    y_prob: np.ndarray,
    all_labels: List[str],
    support: np.ndarray,
    support_cutoff: int,
    t_mama: float,
    t_major: float,
    t_minor: float,
    mama_label_name: str = "Mama",
) -> np.ndarray:
    y_pred = np.zeros_like(y_prob, dtype=int)
    for j, lab in enumerate(all_labels):
        if lab == mama_label_name:
            y_pred[:, j] = (y_prob[:, j] >= t_mama).astype(int)
            continue
        th = t_major if support[j] >= support_cutoff else t_minor
        y_pred[:, j] = (y_prob[:, j] >= th).astype(int)
    return y_pred


# -----------------------------
# Metrics
# -----------------------------
def compute_macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, List[Optional[float]]]:
    aucs: List[Optional[float]] = []
    for j in range(y_true.shape[1]):
        pos = y_true[:, j].sum()
        if pos == 0 or pos == len(y_true):
            aucs.append(None)
            continue
        try:
            aucs.append(float(roc_auc_score(y_true[:, j], y_prob[:, j])))
        except Exception:
            aucs.append(None)
    valid = [a for a in aucs if a is not None]
    return float(np.mean(valid)) if valid else 0.0, aucs


def compute_metrics_panel(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Panel "moderno" (útil para tuning y control de ruido).
    """
    out: Dict[str, float] = {}

    out["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    out["micro_precision"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    out["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))

    out["micro_recall"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    out["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    out["exact_match"] = float(np.mean(np.all(y_true == y_pred, axis=1)))
    out["hamming_loss"] = float(hamming_loss(y_true, y_pred))

    counts = y_pred.sum(axis=1)
    out["pred_labels_mean"] = float(np.mean(counts))
    out["pred_labels_median"] = float(np.median(counts))
    out["pred_labels_p90"] = float(np.percentile(counts, 90))

    macro_auc, _ = compute_macro_auc(y_true, y_prob)
    out["macro_auc"] = float(macro_auc)

    return out


def metrics_like_predict_cascade(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    all_labels: List[str]
) -> Dict[str, Any]:
    """
    Métricas "tipo predict_cascade.py": macro_f1, macro_auc y métricas por etiqueta.
    """
    per_label = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    aucs: List[Optional[float]] = []
    for j in range(y_true.shape[1]):
        pos = y_true[:, j].sum()
        if pos == 0 or pos == len(y_true):
            aucs.append(None)
            continue
        try:
            aucs.append(float(roc_auc_score(y_true[:, j], y_prob[:, j])))
        except Exception:
            aucs.append(None)

    valid = [a for a in aucs if a is not None]
    macro_auc = float(np.mean(valid)) if valid else 0.0

    out: Dict[str, Any] = {
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "per_label_precision": per_label[0].tolist(),
        "per_label_recall": per_label[1].tolist(),
        "per_label_f1": per_label[2].tolist(),
        "per_label_support": per_label[3].tolist(),
        "per_label_auc": aucs,
        "todas_etiquetas": all_labels
    }

    por_etiqueta: Dict[str, Any] = {}
    for j, lab in enumerate(all_labels):
        por_etiqueta[lab] = {
            "precision": out["per_label_precision"][j],
            "recall": out["per_label_recall"][j],
            "f1": out["per_label_f1"][j],
            "support": int(out["per_label_support"][j]),
            "auc": out["per_label_auc"][j],
        }
    out["por_etiqueta"] = por_etiqueta
    return out



def format_predictions_strings(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    all_labels: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Devuelve:
      - PREDICCION_CASCADA: "A; B; C" (orden alfabético)
      - PROBABILIDADES: "A=0.1234; B=0.5678" (mismo orden)
    """
    pred_strs: List[str] = []
    prob_strs: List[str] = []
    idx_map = {lab: j for j, lab in enumerate(all_labels)}

    for i in range(y_pred.shape[0]):
        idxs = np.where(y_pred[i] == 1)[0].tolist()
        labs = [all_labels[j] for j in idxs]
        labs_sorted = sorted(labs)

        if not labs_sorted:
            pred_strs.append("")
            prob_strs.append("")
            continue

        pred_strs.append("; ".join(labs_sorted))
        parts = []
        for lab in labs_sorted:
            j = idx_map[lab]
            parts.append(f"{lab}={float(y_prob[i, j]):.4f}")
        prob_strs.append("; ".join(parts))

    return pred_strs, prob_strs



def pick_best(
    candidates: List[Dict[str, Any]],
    select_metric: str,
    max_p90_labels: Optional[float] = None,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    best_score = -1e18

    for c in candidates:
        m = c["metrics"]
        if max_p90_labels is not None and m.get("pred_labels_p90", 1e9) > max_p90_labels:
            c["constraint_failed"] = True
            continue
        c["constraint_failed"] = False

        score = float(m.get(select_metric, -1e18))
        if best is None or score > best_score:
            best = c
            best_score = score

    if best is None:
        raise RuntimeError("Ningún candidato cumple la restricción max_p90_labels (o métrica inexistente).")
    best["best_score"] = best_score
    return best



def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML de configuración")

    # Overrides opcionales
    ap.add_argument("--mode", choices=["tune", "test"], default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--select-metric", choices=["macro_f1", "micro_f1", "micro_recall", "macro_recall", "exact_match"], default=None)
    ap.add_argument("--max-p90-labels", type=float, default=None)
    return ap


def merge_cfg(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.csv is not None:
        cfg["csv"] = args.csv
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.select_metric is not None:
        cfg["select_metric"] = args.select_metric
    if args.max_p90_labels is not None:
        cfg["max_p90_labels"] = args.max_p90_labels
    return cfg



def main():
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    cfg = merge_cfg(cfg, args)

    required = [
        "mode", "csv", "sep", "text_col", "labels_col", "outdir",
        "path_mama_model", "path_mama_labels",
        "path_resto_model", "path_resto_labels",
        "threshold_mama",
    ]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Falta clave obligatoria en YAML: {k}")

    mode = cfg["mode"]
    csv_path = cfg["csv"]
    sep = cfg.get("sep", ";")
    text_col = cfg.get("text_col", "TEXTO")
    labels_col = cfg.get("labels_col", "NEOPLASIAS")
    outdir = cfg.get("outdir", "output/cascade_tuning")
    select_metric = cfg.get("select_metric", "macro_f1")
    max_p90_labels = cfg.get("max_p90_labels", None)

    best_threshold_json = cfg.get("best_threshold_json", "best_threshold.json")
    mama_label_name = cfg.get("mama_label_name", "Mama")

    thresholds_resto = cfg.get("thresholds_resto", [0.5, 0.3, 0.2])
    two_level = bool(cfg.get("two_level", False))
    support_cutoff = int(cfg.get("support_cutoff", 10))
    thresholds_major = cfg.get("thresholds_major", [0.5, 0.4, 0.3])
    thresholds_minor = cfg.get("thresholds_minor", [0.3, 0.25, 0.2])

    batch_size = int(cfg.get("batch_size", 16))
    max_len = int(cfg.get("max_len", 512))
    device_str = cfg.get("device", None)
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path, sep=sep)
    if text_col not in df.columns:
        raise ValueError(f"No existe columna {text_col} en {csv_path}")
    if labels_col not in df.columns:
        raise ValueError(f"No existe columna {labels_col} en {csv_path}")
    texts = df[text_col].astype(str).tolist()

    all_labels, all_label2idx, _, _, mama_idx2label, resto_idx2label = build_union_labelspace(
        cfg["path_mama_labels"], cfg["path_resto_labels"]
    )

    reverse_syn_map = load_synonyms_reverse_map(cfg["path_resto_labels"], all_labels)
    canonical_set = set(all_labels)

    y_true = build_y_true(
        df, labels_col, all_label2idx, sep=sep,
        reverse_syn_map=reverse_syn_map, canonical_set=canonical_set
    )
    support = compute_support(y_true)

    tok_mama, mod_mama = load_model(cfg["path_mama_model"], device)
    tok_resto, mod_resto = load_model(cfg["path_resto_model"], device)

    probs_mama_local = predict_probs_batch(texts, tok_mama, mod_mama, device=device, batch_size=batch_size, max_len=max_len)
    probs_resto_local = predict_probs_batch(texts, tok_resto, mod_resto, device=device, batch_size=batch_size, max_len=max_len)

    y_prob = combine_probs(
        probs_mama_local, probs_resto_local,
        mama_idx2label, resto_idx2label,
        all_labels, all_label2idx
    )

    t_mama = float(cfg["threshold_mama"])

    if mode == "tune":
        candidates: List[Dict[str, Any]] = []

        if not two_level:
            for t_resto in thresholds_resto:
                t_resto = float(t_resto)
                y_pred = apply_thresholds_global(y_prob, all_labels, t_mama=t_mama, t_resto=t_resto, mama_label_name=mama_label_name)
                metrics = compute_metrics_panel(y_true, y_prob, y_pred)
                candidates.append({
                    "type": "global",
                    "thresholds": {"t_mama": t_mama, "t_resto": t_resto},
                    "metrics": metrics,
                })
        else:
            for t_major in thresholds_major:
                for t_minor in thresholds_minor:
                    t_major = float(t_major)
                    t_minor = float(t_minor)
                    y_pred = apply_thresholds_two_level(
                        y_prob, all_labels, support=support, support_cutoff=support_cutoff,
                        t_mama=t_mama, t_major=t_major, t_minor=t_minor, mama_label_name=mama_label_name
                    )
                    metrics = compute_metrics_panel(y_true, y_prob, y_pred)
                    candidates.append({
                        "type": "two_level",
                        "support_cutoff": support_cutoff,
                        "thresholds": {"t_mama": t_mama, "t_major": t_major, "t_minor": t_minor},
                        "metrics": metrics,
                    })

        Path(outdir, "metrics_by_threshold.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        best = pick_best(candidates, select_metric=select_metric, max_p90_labels=max_p90_labels)

        best_out = {
            "selected_on": "validation",
            "select_metric": select_metric,
            "max_p90_labels": max_p90_labels,
            "best_type": best["type"],
            "best_thresholds": best["thresholds"],
            "best_metrics": best["metrics"],
            "support_cutoff": best.get("support_cutoff", support_cutoff),
            "note": "Umbral seleccionado en VALIDACIÓN interna. No usar TEST para esta selección."
        }
        Path(outdir, best_threshold_json).write_text(
            json.dumps(best_out, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("✓ Tuning completado en VALIDACIÓN")
        print(json.dumps(best_out, ensure_ascii=False, indent=2))

    elif mode == "test":
        best_path = Path(outdir, best_threshold_json)
        if not best_path.exists():
            raise FileNotFoundError(f"No existe {best_path}. Ejecuta primero mode=tune en VALIDACIÓN.")
        best = json.loads(best_path.read_text(encoding="utf-8"))

        best_type = best["best_type"]
        th = best["best_thresholds"]

        if best_type == "global":
            y_pred = apply_thresholds_global(
                y_prob, all_labels,
                t_mama=float(th["t_mama"]),
                t_resto=float(th["t_resto"]),
                mama_label_name=mama_label_name
            )
        else:
            y_pred = apply_thresholds_two_level(
                y_prob, all_labels,
                support=support,
                support_cutoff=int(best.get("support_cutoff", cfg.get("support_cutoff", 10))),
                t_mama=float(th["t_mama"]),
                t_major=float(th["t_major"]),
                t_minor=float(th["t_minor"]),
                mama_label_name=mama_label_name
            )

        metrics = compute_metrics_panel(y_true, y_prob, y_pred)
        Path(outdir, "test_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        pred_strs, prob_strs = format_predictions_strings(y_prob, y_pred, all_labels)
        df_out = pd.DataFrame({
            labels_col: df[labels_col].values,
            "PREDICCION_CASCADA": pred_strs,
            "PROBABILIDADES": prob_strs
        })
        df_out.to_csv(Path(outdir) / "resultados_cascada.csv", index=False, sep=sep)

        old_style = metrics_like_predict_cascade(y_true, y_prob, y_pred, all_labels)
        Path(outdir, "resultados_cascada_metricas.json").write_text(
            json.dumps(old_style, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print("Evaluación en TEST completada")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Guardado CSV: {Path(outdir) / 'resultados_cascada.csv'}")
        print(f" Guardado métricas estilo antiguo: {Path(outdir) / 'resultados_cascada_metricas.json'}")

    else:
        raise ValueError("mode debe ser 'tune' o 'test'.")


if __name__ == "__main__":
    main()