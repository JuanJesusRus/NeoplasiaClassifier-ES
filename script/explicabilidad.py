import torch
import pandas as pd
import yaml
import ast
import os
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from lime.lime_text import LimeTextExplainer
from pathlib import Path


with open("C:\\Users\\jesus\\OneDrive - Universidad de Málaga\\Cuarto\\TFG\\NeoplasiaClassifier-ES\\config_lime.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


Path(config["salida_dir"]).mkdir(parents=True, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(config["ruta_modelo"])
model_config = AutoConfig.from_pretrained(config["ruta_modelo"])
model = AutoModelForSequenceClassification.from_config(model_config)
state_dict = load_file(config["archivo_pesos"])
model.load_state_dict(state_dict)
model.to("cpu")
model.eval()


def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()


df = pd.read_csv(config["ruta_csv"], sep=";", encoding="utf-8-sig")
df["neoplasias_list"] = df[config["columna_combinacion"]].apply(ast.literal_eval)
'''
# Normalizar combinación objetivo (para comparar sin importar el orden)
comb_obj = sorted(config["combinacion_objetivo"])
def coincide_combinacion(row_list):
    return sorted(row_list) == comb_obj

df_filtrado = df[df["neoplasias_list"].apply(coincide_combinacion)].copy()
print(df_filtrado)

if df_filtrado.empty:
    print("No se encontraron textos con esa combinación.")
    exit()

'''

cualquiera = bool(config.get("cualquiera", False))

comb_obj = sorted(config["combinacion_objetivo"])
target_set = set(comb_obj)

def coincide_combinacion(row_list):
    row_set = set(row_list)
    if cualquiera:
        
        return target_set.issubset(row_set)
    else:
        return row_set == target_set

df_filtrado = df[df["neoplasias_list"].apply(coincide_combinacion)].copy()
print(df_filtrado)

if df_filtrado.empty:
    print("No se encontraron textos con esa combinación.")
    exit()
    
seleccionado = int(input())

explainer = LimeTextExplainer(class_names=["Una neoplasia", "Múltiples neoplasias"])

for i, row in df_filtrado.iterrows():
    if i == seleccionado:
        texto = row[config["columna_texto"]]
        etiqueta = row[config["columna_etiqueta"]]
        nombre_archivo = f"{'_'.join(comb_obj)}_{i}.html".replace(" ", "_")
        salida_path = os.path.join(config["salida_dir"], nombre_archivo)

        print(f"Explicando texto índice {i}...")

        exp = explainer.explain_instance(texto, predict_proba, num_features=20)
        exp.save_to_file(f"{salida_path}_{comb_obj}")  

    

