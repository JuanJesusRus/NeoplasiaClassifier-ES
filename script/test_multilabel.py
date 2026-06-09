import ast
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns




from train_multilabel import (
    normalize_token_to_canonical,
    parse_collections_file,
    filter_labels_exclude_mama,
    save_confusion_matrices as save_confusion_matrices_train
)

label2idx = {}
reverse_syn_map = {}


def load_label_map(model_dir: Path) -> dict:
    candidates = [
        model_dir / "label2idx.json",
        model_dir.parent / "label2idx.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                return json.loads(path.read_text(encoding="latin-1"))
    raise FileNotFoundError(
        f"No se encontró label2idx.json en {candidates[0]} ni en {candidates[1]}"
    )


def build_reverse_syn_map(labels_list, labels_file=None, synonyms_path=None, case_sensitive=True):
    synonyms_map = {}

    if labels_file:
        labels_from_file, synonyms_map = parse_collections_file(str(labels_file))
        if not labels_list:
            labels_list = list(labels_from_file)
    elif synonyms_path and synonyms_path.exists():
        try:
            synonyms_map = json.loads(synonyms_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            synonyms_map = json.loads(synonyms_path.read_text(encoding="latin-1"))

    rev_map = {}
    for canon, syns in synonyms_map.items():
        key = canon if case_sensitive else canon.lower()
        rev_map[key] = canon
        for syn in syns:
            skey = syn if case_sensitive else syn.lower()
            rev_map[skey] = canon

    for label in labels_list:
        key = label if case_sensitive else label.lower()
        rev_map[key] = label

    return rev_map



class MultilabelDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        labels_list,
        max_length=512,
        text_col="TEXTO",
        labels_col="NEOPLASIAS",
        sep=";",
        unknown_policy="other",
        labels_case_sensitive=True
    ):
        self.df = df.reset_index(drop=True)
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.labels_col = labels_col
        self.sep = sep
        self.unknown_policy = unknown_policy
        self.labels_case_sensitive = labels_case_sensitive

    def __len__(self):
        return len(self.df)

    def _get_label_vector(self, idx):
        raw = self.df.iloc[idx][self.labels_col]
        if pd.isna(raw) or str(raw).strip() == "":
            return torch.zeros(len(self.labels_list), dtype=torch.float)

        if isinstance(raw, list):
            parts = [str(x).strip() for x in raw if str(x).strip()]
        else:
            s = str(raw).strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parts = ast.literal_eval(s)
                    if isinstance(parts, str):
                        parts = [p.strip() for p in parts.split(self.sep) if p.strip()]
                    else:
                        parts = [str(p).strip() for p in parts if str(p).strip()]
                except Exception:
                    parts = [p.strip() for p in s.strip("[]").split(self.sep) if p.strip()]
            else:
                parts = [p.strip() for p in s.split(self.sep) if p.strip()]

        vec = np.zeros(len(self.labels_list), dtype=float)
        for p in parts:
            canon = normalize_token_to_canonical(
                p,
                reverse_syn_map,
                set(self.labels_list),
                case_sensitive=self.labels_case_sensitive
            )
            if canon:
                vec[label2idx[canon]] = 1.0
            elif self.unknown_policy == "other" and "Otro" in label2idx:
                vec[label2idx["Otro"]] = 1.0
        return torch.tensor(vec, dtype=torch.float)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.df.iloc[idx][self.text_col]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = self._get_label_vector(idx)
        return item




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

    per_label = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)

    aucs = []
    valid_aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc_i = roc_auc_score(y_true[:, i], y_prob[:, i])
                aucs.append(auc_i)
                valid_aucs.append(auc_i)
            except ValueError:
                aucs.append(None)
        else:
            aucs.append(None)

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None

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
        "per_label": per_label,
        "aucs": aucs,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "hamming_loss": hamming
    }



def save_confusion_matrices(y_true, y_pred, labels_list, out_dir):
    save_confusion_matrices_train(y_true, y_pred, labels_list, out_dir)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--text-col", type=str, default="TEXTO")
    parser.add_argument("--labels-col", type=str, default="NEOPLASIAS")
    parser.add_argument("--sep", type=str, default=";")
    parser.add_argument("--labels-file", type=str, default=None)
    parser.add_argument("--unknown-policy", choices=["ignore", "other", "error"], default="other")
    parser.add_argument("--labels-case-sensitive", dest="labels_case_sensitive", action="store_true")
    parser.add_argument("--no-labels-case-sensitive", dest="labels_case_sensitive", action="store_false")
    parser.set_defaults(labels_case_sensitive=True)
    parser.add_argument("--filter-exclude-mama", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global label2idx, reverse_syn_map

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    df = pd.read_csv(args.test_csv, sep=args.sep, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df = filter_labels_exclude_mama(
        df,
        labels_col=args.labels_col,
        sep=args.sep,
        enable=args.filter_exclude_mama
    )

    label2idx = load_label_map(model_dir)
    labels_list = [None] * len(label2idx)
    for label, idx in label2idx.items():
        labels_list[idx] = label

    labels_file = Path(args.labels_file) if args.labels_file else None
    synonyms_path = model_dir / "synonyms_map.json"
    if not synonyms_path.exists():
        synonyms_path = model_dir.parent / "synonyms_map.json"
    reverse_syn_map = build_reverse_syn_map(
        labels_list,
        labels_file=labels_file,
        synonyms_path=synonyms_path if synonyms_path.exists() else None,
        case_sensitive=args.labels_case_sensitive
    )

    dataset = MultilabelDataset(
        df,
        tokenizer,
        labels_list,
        max_length=args.max_len,
        text_col=args.text_col,
        labels_col=args.labels_col,
        sep=args.sep,
        unknown_policy=args.unknown_policy,
        labels_case_sensitive=args.labels_case_sensitive
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate(model, loader, device, args.threshold)

    with open(out_dir / "test_metrics.json", "w", encoding="utf8") as f:
        json.dump(
            {
                "macro_auc": metrics["macro_auc"],
                "micro_auc": metrics["micro_auc"],
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "macro_precision": metrics["macro_precision"],
                "micro_precision": metrics["micro_precision"],
                "macro_recall": metrics["macro_recall"],
                "micro_recall": metrics["micro_recall"],
                "hamming_loss": metrics["hamming_loss"],
                "per_label_precision": metrics["per_label"][0].tolist(),
                "per_label_recall": metrics["per_label"][1].tolist(),
                "per_label_f1": metrics["per_label"][2].tolist(),
                "per_label_support": metrics["per_label"][3].tolist(),
                "per_label_auc": metrics["aucs"],
                "labels": labels_list,
            },
            f,
            indent=4
        )

    save_confusion_matrices(
        metrics["y_true"],
        metrics["y_pred"],
        labels_list,
        out_dir / "test_results"
    )

    print("\n=== TEST RESULTS ===")
    print(f"Macro-AUC: {metrics['macro_auc'] if metrics['macro_auc'] is not None else 'N/A'}")
    print(f"Micro-AUC: {metrics['micro_auc'] if metrics['micro_auc'] is not None else 'N/A'}")
    print(f"Macro-F1 : {metrics['macro_f1']:.4f}")
    print(f"Micro-F1 : {metrics['micro_f1']:.4f}")
    print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
    print(f"Micro-Precision: {metrics['micro_precision']:.4f}")
    print(f"Macro-Recall: {metrics['macro_recall']:.4f}")
    print(f"Micro-Recall: {metrics['micro_recall']:.4f}")
    print(f"Hamming-Loss: {metrics['hamming_loss']:.4f}")


if __name__ == "__main__":
    main()
