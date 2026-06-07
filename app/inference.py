import os
import json
import unicodedata
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime_utils import explicar_texto

class NeoplasiaInference:
    def __init__(self, config):
        # Obtener directorio base (donde está la app)
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Configuración del modelo binario
        self.ruta_modelo = config["modelo"]["ruta_modelo"]
        self.ruta_tokenizer = config["modelo"]["ruta_tokenizer"]
        self.max_length = config["modelo"]["max_length"]

        # Cargar tokenizer y modelo binario
        self.tokenizer = AutoTokenizer.from_pretrained(self.ruta_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.ruta_modelo)
        self.model.eval()

        # Mapeo de clases: 0 = una neoplasia, 1 = múltiples
        self.id2label = {0: "Una neoplasia", 1: "Múltiples neoplasias"}

        # Configuración de cascade opcional
        cascade_config = config.get("cascade", {})
        self.cascade_enabled = all(
            key in cascade_config
            for key in [
                "ruta_mama_model",
                "ruta_mama_labels",
                "ruta_resto_model",
                "ruta_resto_labels"
            ]
        )

        if self.cascade_enabled:
            # Resolver rutas relativas
            self.ruta_mama_model = self._resolve_path(cascade_config["ruta_mama_model"])
            self.ruta_mama_labels = self._resolve_path(cascade_config["ruta_mama_labels"])
            self.ruta_resto_model = self._resolve_path(cascade_config["ruta_resto_model"])
            self.ruta_resto_labels = self._resolve_path(cascade_config["ruta_resto_labels"])
            self.umbral_mama = cascade_config.get("umbral_mama", 0.5)
            self.umbral_resto = cascade_config.get("umbral_resto", 0.5)

            print(f"  Mama model: {self.ruta_mama_model}")
            print(f"  Resto model: {self.ruta_resto_model}")

            self.mama_tokenizer = AutoTokenizer.from_pretrained(self.ruta_mama_model)
            self.mama_model = AutoModelForSequenceClassification.from_pretrained(self.ruta_mama_model)
            self.mama_model.eval()

            self.resto_tokenizer = AutoTokenizer.from_pretrained(self.ruta_resto_model)
            self.resto_model = AutoModelForSequenceClassification.from_pretrained(self.ruta_resto_model)
            self.resto_model.eval()

            self.id2label_mama = self._load_label_map(self.ruta_mama_labels)
            self.id2label_resto = self._load_label_map(self.ruta_resto_labels)
            
        else:
            self.ruta_mama_model = None
            self.ruta_mama_labels = None
            self.ruta_resto_model = None
            self.ruta_resto_labels = None
            self.umbral_mama = 0.5
            self.umbral_resto = 0.5
            self.mama_tokenizer = None
            self.mama_model = None
            self.resto_tokenizer = None
            self.resto_model = None
            self.id2label_mama = {}
            self.id2label_resto = {}

    def normalizar(self, texto):
        texto = texto.lower()
        texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def _resolve_path(self, path):
        """Resuelve rutas relativas basándose en el directorio de la app."""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.app_dir, path))

    def _load_label_map(self, labels_dir):
        label2idx_path = os.path.join(labels_dir, "label2idx.json")
        
        if not os.path.exists(label2idx_path):
            print(f"ERROR: No encontrado {label2idx_path}")
            print(f"  Directorio buscado: {labels_dir}")
            print(f"  Archivos disponibles: {os.listdir(labels_dir) if os.path.exists(labels_dir) else 'DIR NO EXISTE'}")
            return {}
            
        try:
            with open(label2idx_path, "r", encoding="utf-8") as f:
                label2idx = json.load(f)
        except UnicodeDecodeError:
            with open(label2idx_path, "r", encoding="latin-1") as f:
                label2idx = json.load(f)
        except Exception as e:
            print(f"ERROR cargando {label2idx_path}: {e}")
            return {}

        return {int(idx): label for label, idx in label2idx.items()}


    def _predict_multilabel(self, texto, tokenizer, model, id2label, threshold):
        inputs = tokenizer(
            texto,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]

        predictions = {}
        for idx, prob in enumerate(probs):
            if prob >= threshold and idx in id2label:
                predictions[id2label[idx]] = float(round(float(prob), 4))

        # Devolver también las probabilidades completas
        all_probs = {id2label.get(idx, f"idx_{idx}"): float(round(float(prob), 4)) 
                    for idx, prob in enumerate(probs)}
        
        return predictions, all_probs

    def _cascade_predict(self, texto):
        if not self.cascade_enabled:
            return {
                "activado": False,
                "predicciones": {},
                "error": "Cascade no está habilitado",
                "umbral_mama": self.umbral_mama,
                "umbral_resto": self.umbral_resto
            }

        # Verificar si se cargaron las etiquetas
        if not self.id2label_mama or not self.id2label_resto:
            return {
                "activado": False,
                "predicciones": {},
                "error": f"Etiquetas no cargadas. Mama: {len(self.id2label_mama)}, Resto: {len(self.id2label_resto)}",
                "umbral_mama": self.umbral_mama,
                "umbral_resto": self.umbral_resto
            }

        mama_preds, mama_all_probs = self._predict_multilabel(
            texto,
            self.mama_tokenizer,
            self.mama_model,
            self.id2label_mama,
            self.umbral_mama,
        )

        resto_preds, resto_all_probs = self._predict_multilabel(
            texto,
            self.resto_tokenizer,
            self.resto_model,
            self.id2label_resto,
            self.umbral_resto,
        )

        all_labels = list(self.id2label_mama.values()) + [label for label in self.id2label_resto.values() if label not in self.id2label_mama.values()]
        combined_preds = {}

        for label in all_labels:
            prob_mama = mama_preds.get(label, 0.0)
            prob_resto = resto_preds.get(label, 0.0)
            max_prob = max(prob_mama, prob_resto)
            if max_prob > 0:
                combined_preds[label] = float(round(max_prob, 4))

        return {
            "activado": True,
            "predicciones": dict(sorted(combined_preds.items())),
            "probabilidades_mama": mama_all_probs,
            "probabilidades_resto": resto_all_probs,
            "umbral_mama": self.umbral_mama,
            "umbral_resto": self.umbral_resto
        }

    def predecir(self, texto: str, mode: str = "binario", generar_explicacion: bool = False):
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

        resultado = {
            "prediccion_binaria": pred_idx,
            "clase": pred_label,
            "probabilidad": round(score, 4)
        }

        if mode == "binario" and generar_explicacion:
            resultado["explicacion"] = explicar_texto(texto, self.model, self.tokenizer, self.max_length)

        if mode == "multietiqueta":
            resultado["multietiqueta"] = self._cascade_predict(texto)

        return resultado
