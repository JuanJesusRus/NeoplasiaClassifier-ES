import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score
import json
try:
    from script.train_multilabel import normalize_token_to_canonical, normalize_token_variations
except Exception:
    def normalize_token_variations(x):
        return x
    def normalize_token_to_canonical(token, reverse_syn_map, canonical_set, case_sensitive=True):
        t = token.strip()
        if not t:
            return None
        if not case_sensitive:
            for c in canonical_set:
                if t.lower() == c.lower():
                    return c
        else:
            if t in canonical_set:
                return t
        key = t.lower() if not case_sensitive else t
        return reverse_syn_map.get(key)

# === CONFIGURACIÓN ===
PATH_MAMA_MODEL = "C:\\Users\\jesus\\OneDrive - Universidad de Málaga\\Cuarto\\TFG\\NeoplasiaClassifier-ES\\output\\multilabel_modelo_mama\\best_model"
PATH_MAMA_LABELS = "C:\\Users\\jesus\\OneDrive - Universidad de Málaga\\Cuarto\\TFG\\NeoplasiaClassifier-ES\\output\\multilabel_modelo_mama"
PATH_RESTO_MODEL = "C:\\Users\\jesus\\OneDrive - Universidad de Málaga\\Cuarto\\TFG\\NeoplasiaClassifier-ES\\output\\multilabel_modelo_resto\\best_model"
PATH_RESTO_LABELS = "C:\\Users\\jesus\\OneDrive - Universidad de Málaga\\Cuarto\\TFG\\NeoplasiaClassifier-ES\\output\\multilabel_modelo_resto"
CSV_TEST = "output/comparacionModelos/datos/test_set_completo_cambiado.csv"
OUTPUT_FILE = "output/multilabel_cascade/resultados_cascada.csv"

UMBRAL_MAMA = 0.5  
UMBRAL_RESTO = 0.5 

def obtener_etiquetas_reales(csv_path):
    """Obtiene etiquetas reales si existen en el CSV."""
    df = pd.read_csv(csv_path, sep=";")
    if "LABELS" in df.columns or "labels" in df.columns:
        col = "LABELS" if "LABELS" in df.columns else "labels"
        return df[col].tolist()
    return None


def calcular_metricas(y_true_list, predicciones_list, todas_etiquetas, threshold=0.5):
    """Calcula métricas similares a train_multilabel.
    
    Args:
        y_true_list: lista de strings con etiquetas reales (ej: "Mama;Sarcoma")
        predicciones_list: lista de arrays de probabilidades por paciente
        todas_etiquetas: lista de todas las posibles etiquetas
        threshold: umbral para convertir probabilidades a predicción
    
    Returns:
        dict con métricas (macro_f1, macro_auc, per_label, etc)
    """
    label2idx = {label: i for i, label in enumerate(todas_etiquetas)}
    
    y_true = np.zeros((len(y_true_list), len(todas_etiquetas)))
    y_prob = np.zeros((len(predicciones_list), len(todas_etiquetas)))
    
    for i, labels_str in enumerate(y_true_list):
        if labels_str and str(labels_str).strip():
            labels = [l.strip() for l in str(labels_str).split(";")]
            for label in labels:
                if label in label2idx:
                    y_true[i, label2idx[label]] = 1
    
    for i, probs in enumerate(predicciones_list):
        y_prob[i, :] = probs
    
    y_pred = (y_prob >= threshold).astype(int)
    
    per_label = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            try:
                aucs.append(float(roc_auc_score(y_true[:, i], y_prob[:, i])))
            except Exception:
                aucs.append(None)
        else:
            aucs.append(None)
    
    valid_aucs = [a for a in aucs if a is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    
    return {
        "macro_f1": float(macro_f1),
        "macro_auc": macro_auc,
        "per_label_precision": per_label[0].tolist(),
        "per_label_recall": per_label[1].tolist(),
        "per_label_f1": per_label[2].tolist(),
        "per_label_support": per_label[3].tolist(),
        "per_label_auc": aucs,
        "todas_etiquetas": todas_etiquetas
    }


def cargar_modelo(path):
    """Carga modelo y tokenizer desde una ruta."""
    if not os.path.exists(path):
        print(f"ERROR: No existe la carpeta {path}. ¿Has ejecutado el entrenamiento?")
        exit()
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval() 
        return tokenizer, model
    except Exception as e:
        print(f"ERROR cargando {path}: {e}")
        exit()


def cargar_label_map(path):
    """Carga el mapeo de etiquetas desde label2idx.json.
    
    Returns:
        dict: {idx: label_name} invertido desde label2idx.json
    """
    label2idx_path = os.path.join(path, "label2idx.json")
    if os.path.exists(label2idx_path):
        try:
            with open(label2idx_path, 'r', encoding='utf-8') as f:
                label2idx = json.load(f)
        except UnicodeDecodeError:
            with open(label2idx_path, 'r', encoding='latin-1') as f:
                label2idx = json.load(f)
        
        idx2label = {int(v): k for k, v in label2idx.items()}
        print(f"   Cargadas {len(idx2label)} etiquetas desde {label2idx_path}")
        return idx2label
    else:
        print(f"ERROR: No encontramos label2idx.json en {path}")
        print(f"       Path buscado: {label2idx_path}")
        return {}

def predecir(texto, tokenizer, model, id2label_correct, threshold):
    """Predice etiquetas y devuelve probabilidades.
    
    Args:
        id2label_correct: mapeo correcto de índice a nombre (desde label2idx.json)
    
    Returns:
        dict con:
        - 'predicciones': dict etiqueta -> probabilidad (solo las que superan threshold)
        - 'todas_probs': array de probabilidades para TODAS las etiquetas
    """
    inputs = tokenizer(str(texto), return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]
    
    predicciones = {}  
    
    for idx, prob in enumerate(probs):
        if prob >= threshold:
            if idx in id2label_correct:
                etiqueta = id2label_correct[idx]
                predicciones[etiqueta] = float(prob)
    
    return {
        'predicciones': predicciones,
        'todas_probs': probs,
    }

def main():
    print("--- INICIANDO PREDICCIÓN EN CASCADA ---")
    
    print("1. Cargando modelos...")
    tok_mama, mod_mama = cargar_modelo(PATH_MAMA_MODEL)
    tok_resto, mod_resto = cargar_modelo(PATH_RESTO_MODEL)
    
    print("   Cargando mapeos de etiquetas...")
    id2label_mama = cargar_label_map(PATH_MAMA_LABELS)
    id2label_resto = cargar_label_map(PATH_RESTO_LABELS)

    reverse_syn_map = {}
    syn_path = os.path.join(PATH_RESTO_LABELS, "synonyms_map.json")
    if os.path.exists(syn_path):
        try:
            with open(syn_path, 'r', encoding='utf-8') as f:
                syn_map = json.load(f)
        except UnicodeDecodeError:
            with open(syn_path, 'r', encoding='latin-1') as f:
                syn_map = json.load(f)
        # Build reverse map (lowercase keys)
        for canon, syns in syn_map.items():
            reverse_syn_map[canon.lower()] = canon
            for s in syns:
                reverse_syn_map[s.lower()] = canon
        for v in list(id2label_resto.values()) + list(id2label_mama.values()):
            reverse_syn_map[str(v).lower()] = v
    
    todas_etiquetas_mama = list(id2label_mama.values())
    todas_etiquetas_resto = list(id2label_resto.values())
    
    todas_etiquetas = todas_etiquetas_mama.copy()  # Empezar con Mama
    for etiqueta in todas_etiquetas_resto:
        if etiqueta not in todas_etiquetas:
            todas_etiquetas.append(etiqueta)
    
    print(f"   Etiquetas Mama ({len(todas_etiquetas_mama)}): {todas_etiquetas_mama}")
    print(f"   Etiquetas Resto ({len(todas_etiquetas_resto)}): {todas_etiquetas_resto[:5]}... ({len(todas_etiquetas_resto)} total)")
    print(f"   Orden final ({len(todas_etiquetas)}): Mama primero, luego resto")
    
    if not os.path.exists(CSV_TEST):
        print(f"ERROR: No encuentro el CSV de test en {CSV_TEST}")
        exit()

    print(f"2. Leyendo dataset: {CSV_TEST}")
    df = pd.read_csv(CSV_TEST, sep=";") 
    
    if "TEXTO" not in df.columns:
        print("ERROR: El CSV no tiene una columna llamada 'TEXTO'")
        exit()
    df["TEXTO"] = df["TEXTO"].astype(str)
    
    predicciones_finales = []
    probabilidades_finales = []
    todas_probs_cascada = []  
    
    print(f"3. Procesando {len(df)} pacientes...")
    print(f"   - Umbral Mama: {UMBRAL_MAMA}")
    print(f"   - Umbral Resto: {UMBRAL_RESTO}")
    
    # 3. Bucle paciente por paciente
    for i, row in tqdm(df.iterrows(), total=len(df)):
        texto = row["TEXTO"]
        
        result_a = predecir(texto, tok_mama, mod_mama, id2label_mama, threshold=UMBRAL_MAMA)
        preds_a = result_a['predicciones']
        
        result_b = predecir(texto, tok_resto, mod_resto, id2label_resto, threshold=UMBRAL_RESTO)
        preds_b = result_b['predicciones']
        
        # --- COMBINAR PROBABILIDADES ---
        # Crear vector de probabilidades cascada (unión de ambos modelos)
        # Basarse en NOMBRES de etiquetas, no en índices locales
        probs_cascada = np.zeros(len(todas_etiquetas))
        for etiqueta_idx, etiqueta in enumerate(todas_etiquetas):
            if etiqueta in preds_a:
                probs_cascada[etiqueta_idx] = max(probs_cascada[etiqueta_idx], preds_a[etiqueta])
            if etiqueta in preds_b:
                probs_cascada[etiqueta_idx] = max(probs_cascada[etiqueta_idx], preds_b[etiqueta])
        todas_probs_cascada.append(probs_cascada)
        
        total_preds = {**preds_a, **preds_b}  
        
        etiquetas_ordenadas = sorted(total_preds.keys())
        if not etiquetas_ordenadas:
            res_str = "" 
            prob_str = ""
        else:
            res_str = "; ".join(etiquetas_ordenadas)
            prob_str = "; ".join([f"{etiqueta}={total_preds[etiqueta]:.4f}" for etiqueta in etiquetas_ordenadas])
            
        predicciones_finales.append(res_str)
        probabilidades_finales.append(prob_str)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df["PREDICCION_CASCADA"] = predicciones_finales
    df["PROBABILIDADES"] = probabilidades_finales

    keep_cols = []
    if "NEOPLASIAS" in df.columns:
        keep_cols.append("NEOPLASIAS")
    if "PREDICCION_CASCADA" in df.columns:
        keep_cols.append("PREDICCION_CASCADA")
    if "PROBABILIDADES" in df.columns:
        keep_cols.append("PROBABILIDADES")

    if keep_cols:
        df_out = df[keep_cols]
    else:
        df_out = pd.DataFrame({
            "PREDICCION_CASCADA": predicciones_finales,
            "PROBABILIDADES": probabilidades_finales
        })

    df_out.to_csv(OUTPUT_FILE, index=False, sep=";")
    
    etiquetas_reales = None
    if "LABELS" in df.columns:
        etiquetas_reales = df["LABELS"].tolist()
    elif "labels" in df.columns:
        etiquetas_reales = df["labels"].tolist()
    elif "NEOPLASIAS" in df.columns:
        etiquetas_reales = df["NEOPLASIAS"].tolist()
    
    if etiquetas_reales:
        print("\n4. Calculando métricas con respecto a etiquetas reales...")
        
        etiquetas_reales_procesadas = []
        import ast
        for item in etiquetas_reales:
            parts = []
            if isinstance(item, str):
                s = item.strip()
                if s.startswith('[') and s.endswith(']'):
                    try:
                        lista = ast.literal_eval(s)
                        if isinstance(lista, list):
                            parts = [str(x).strip() for x in lista]
                        else:
                            parts = [p.strip() for p in s.strip('[]').split(';') if p.strip()]
                    except Exception:
                        parts = [p.strip() for p in s.strip('[]').split(';') if p.strip()]
                else:
                    parts = [p.strip() for p in s.split(';') if p.strip()]
            else:
                parts = [str(item)]

            canons = []
            for p in parts:
                canon = normalize_token_to_canonical(p, reverse_syn_map, set(todas_etiquetas), case_sensitive=False)
                if canon:
                    canons.append(canon)
                else:
                    npart = normalize_token_variations(p)
                    matched = None
                    for c in todas_etiquetas:
                        if npart.lower() == str(c).lower():
                            matched = c
                            break
                    if matched:
                        canons.append(matched)
                    else:
                        canons.append(p)

            etiquetas_reales_procesadas.append(";".join(canons))

        metricas = calcular_metricas(etiquetas_reales_procesadas, todas_probs_cascada, todas_etiquetas, threshold=UMBRAL_RESTO)
        
        metricas_por_etiqueta = {}
        for j, etiqueta in enumerate(todas_etiquetas):
            metricas_por_etiqueta[etiqueta] = {
                "precision": metricas['per_label_precision'][j],
                "recall": metricas['per_label_recall'][j],
                "f1": metricas['per_label_f1'][j],
                "support": int(metricas['per_label_support'][j]),
                "auc": metricas['per_label_auc'][j]
            }
        
        metricas['por_etiqueta'] = metricas_por_etiqueta
        
        metricas_file = OUTPUT_FILE.replace(".csv", "_metricas.json")
        with open(metricas_file, "w", encoding="utf-8") as f:
            json.dump(metricas, f, ensure_ascii=False, indent=2)
        
        print(f"   Macro-F1: {metricas['macro_f1']:.4f}")
        print(f"   Macro-AUC: {metricas['macro_auc']:.4f}")
        print(f"   Métricas guardadas en: {metricas_file}")
    else:
        print("\n4. No se encontraron etiquetas reales en el CSV. Saltar cálculo de métricas.")
    
    print(f"\nResultados guardados en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()