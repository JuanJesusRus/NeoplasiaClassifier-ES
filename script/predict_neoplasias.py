import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt
from safetensors.torch import load_file
# Configuración
config = {
    "ruta_csv": r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\comparacionModelos\datos\test2_set.csv",  # conjunto a usar
    "columna_texto": "TEXTO",
    "columna_etiqueta": "MULTIPLES",
    "ruta_modelo": r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\xlm_r_galen\xlm_r_galen_1",  # carpeta con config.json y tokenizer
    "archivo_pesos": r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\xlm_r_galen\xlm_r_galen_1\model.safetensors",
    "batch_size": 16,
    "max_length": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 1. Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(config["ruta_modelo"])

# Cargar config y modelo sin pesos
config_modelo = AutoConfig.from_pretrained(config["ruta_modelo"])
model = AutoModelForSequenceClassification.from_config(config_modelo)

# Cargar pesos desde safetensors
state_dict = load_file(config["archivo_pesos"])
model.load_state_dict(state_dict)

# Enviar a dispositivo
model.to(config["device"])
model.eval()
# 2. Cargar CSV como Dataset
df = pd.read_csv(config["ruta_csv"])
dataset = Dataset.from_pandas(df)

# 3. Tokenización
def preprocess(example):
    return tokenizer(example[config["columna_texto"]], truncation=True, padding="max_length", max_length=config["max_length"])

tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", config["columna_etiqueta"]])

# 4. Evaluación
loader = DataLoader(tokenized_dataset, batch_size=config["batch_size"])
all_preds, all_probs, all_labels = [], [], []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(config["device"])
        attention_mask = batch["attention_mask"].to(config["device"])
        labels = batch[config["columna_etiqueta"]].to(config["device"])

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # probabilidad clase 1
        all_labels.extend(labels.cpu().tolist())

# 5. Métricas con sklearn
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"ROC AUC:   {auc:.4f}")
print("Matriz de confusión:")
print(cm)
