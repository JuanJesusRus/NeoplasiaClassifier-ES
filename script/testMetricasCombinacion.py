import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from safetensors.torch import load_file
from pathlib import Path
from tqdm import tqdm
import ast

# -------------------------------
# Configuración general
# -------------------------------
config = {
    "ruta_csv": r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\comparacionModelos\datos\test_set_completo_cambiado.csv",
    "columna_texto": "TEXTO",
    "columna_etiqueta": "MULTIPLES",
    "columna_combinacion": "NEOPLASIAS",
    "batch_size": 16,
    "max_length": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "out_dir": Path(r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\comparacionModelos\comparacionTestCompletos")
}

# Lista de modelos a evaluar
modelos = [
    {
        "nombre": "xlm_roberta_1",
        "ruta_modelo":  r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\xlm_roberta\xlm_roberta1",
        "archivo_pesos": r"F:\TFG_models\temp_model_robertaxlm\roberta_xlm_1\model.safetensors"
    },
    {
        "nombre": "xlm_roberta_2",
        "ruta_modelo":  r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\xlm_roberta\xlm_roberta2",
        "archivo_pesos": r"F:\TFG_models\temp_model_robertaxlm\roberta_xlm_2\model.safetensors"
    },
    {
        "nombre": "xlm_r_galen_1",
        "ruta_modelo":  r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\xlm_r_galen\xlm_r_galen_1",
        "archivo_pesos": r"F:\TFG_models\temp_model_xlm_galen\galen_1\model.safetensors"
    },

    {
        "nombre": "xlm_r_galen_2",
        "ruta_modelo":  r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\xlm_r_galen\xlm_r_galen_2",
        "archivo_pesos": r"F:\TFG_models\temp_model_xlm_galen\galen_2\model.safetensors"
    },

    {
        "nombre": "roberta2_1",
        "ruta_modelo":  r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\roberta\roberta2_1",
        "archivo_pesos": r"F:\TFG_models\temp_models_roberta\roberta_2_1\model.safetensors"
    },

    
    {
        "nombre": "bio_ehr1",
        "ruta_modelo":  r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\bio_ehr\bio_ehr1",
        "archivo_pesos": r"F:\TFG_models\temp_model_bioehr\bioehr_1\model.safetensors"
    }




]


def clave_ordenada(comb_str):
    try:
        lista = ast.literal_eval(comb_str)
        if isinstance(lista, list):
            return tuple(sorted(lista))  
        return ()
    except Exception:
        return ()
    
# -------------------------------
# Función de evaluación
# -------------------------------

def evaluar_modelo(modelo_info, df_base):
    print(f"\n Evaluando modelo: {modelo_info['nombre']}")
    
    tokenizer = AutoTokenizer.from_pretrained(modelo_info["ruta_modelo"])
    config_modelo = AutoConfig.from_pretrained(modelo_info["ruta_modelo"])
    model = AutoModelForSequenceClassification.from_config(config_modelo)
    
    state_dict = load_file(modelo_info["archivo_pesos"])
    model.load_state_dict(state_dict)
    model.to(config["device"])
    model.eval()
    
    dataset = Dataset.from_pandas(df_base.copy())

    def preprocess(example):
        return tokenizer(example[config["columna_texto"]], truncation=True, padding="max_length", max_length=config["max_length"])
    
    tokenized_dataset = dataset.map(preprocess, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", config["columna_etiqueta"]])

    loader = DataLoader(tokenized_dataset, batch_size=config["batch_size"])
    
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencia"):
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = batch[config["columna_etiqueta"]].to(config["device"])

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    df_resultado = df_base.copy()
    df_resultado["preds"] = all_preds
    df_resultado["probs_clase_1"] = all_probs
    df_resultado["etiqueta_real"] = all_labels


    df_resultado["comb_key"] = df_resultado[config["columna_combinacion"]].apply(clave_ordenada)
    resultados = []

    for comb, df_comb in df_resultado.groupby("comb_key"):
        y_true = df_comb[config["columna_etiqueta"]]
        y_pred = df_comb["preds"]
        y_probs = df_comb["probs_clase_1"]
        
        resultados.append({
            "Combinación": str(comb), 
            "Soporte": len(df_comb),
            "AUC": round(roc_auc_score(y_true, y_probs), 3) if len(set(y_true)) > 1 else None,
            "Accuracy": round(accuracy_score(y_true, y_pred), 3),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "F1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "MCC": round(matthews_corrcoef(y_true, y_pred), 3)
        })

    df_metricas = pd.DataFrame(resultados).sort_values(by="Soporte", ascending=False)


    df_metricas["num_neoplasias"] = df_metricas["Combinación"].apply(lambda x: len(ast.literal_eval(x)))

    df_metricas = df_metricas[df_metricas["num_neoplasias"] > 1].copy()


    salida_csv = config["out_dir"] / f"metricas_por_combinacion_{modelo_info['nombre']}.csv"
    df_metricas.to_csv(salida_csv, index=False, encoding="utf-8-sig")
    print(f" Métricas de múltiples neoplasias guardadas en: {salida_csv.name}")


# -------------------------------
# EJECUTAR PARA TODOS LOS MODELOS
# -------------------------------
df_test = pd.read_csv(config["ruta_csv"], sep=";", quotechar='"', engine="python")
for modelo in modelos:
    evaluar_modelo(modelo, df_test)



print("\n Combinando todos los resultados en una única tabla...")

csvs_generados = list(config["out_dir"].glob("metricas_por_combinacion_*.csv"))
df_final = pd.DataFrame()

for path_csv in csvs_generados:
    modelo_nombre = path_csv.stem.replace("metricas_por_combinacion_", "")
    df_tmp = pd.read_csv(path_csv)
    df_tmp["Modelo"] = modelo_nombre
    df_final = pd.concat([df_final, df_tmp], ignore_index=True)

cols = ["Modelo", "Combinación", "Soporte", "AUC", "Accuracy", "Precision", "Recall", "F1", "MCC"]

df_final = df_final.sort_values(by="Combinación").reset_index(drop=True)


ruta_final = config["out_dir"] / "metricas_por_combinacion_TODOS_LOS_MODELOS.csv"
df_final.to_csv(ruta_final, index=False)
print(f" Tabla combinada guardada como: {ruta_final.name}")
