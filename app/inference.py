import unicodedata
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NeoplasiaInference:
    def __init__(self, config):
        # Configuración del modelo
        self.ruta_modelo = config["modelo"]["ruta_modelo"]
        self.ruta_tokenizer = config["modelo"]["ruta_tokenizer"]
        self.max_length = config["modelo"]["max_length"]

        # Cargar tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(self.ruta_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.ruta_modelo)
        self.model.eval()

        # Mapeo de clases: 0 = una neoplasia, 1 = múltiples
        self.id2label = {0: "Una neoplasia", 1: "Múltiples neoplasias"}

    def normalizar(self, texto):
        texto = texto.lower()
        texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto


    def predecir(self, texto: str):
        texto = self.normalizar(texto)

        inputs = self.tokenizer(
            texto,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

        pred_idx = int(probs.argmax())
        pred_label = self.id2label[pred_idx]
        score = float(probs[pred_idx])

        return {
            "prediccion_binaria": pred_idx,
            "clase": pred_label,
            "probabilidad": round(score, 4)
        }
