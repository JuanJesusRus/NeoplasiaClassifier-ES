import pandas as pd
import re
import unicodedata
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    return texto


def cargar_colecciones2(path_archivo):
    """
    Cargamos el .txt y lo convertimos con exec a un diccionario con el que poder trabajar.
    Normalizamos todos los términos de la colección.
    """
    namespace = {}
    with open(path_archivo, "r", encoding="utf-8") as f:
        exec(f.read(), {}, namespace)
    
    colecciones = namespace["colecciones"]
    colecciones_normalizadas = {neo: [normalizar(termino) for termino in terminos] 
                                 for neo, terminos in colecciones.items()}
    return colecciones_normalizadas

def cargar_colecciones(path_archivo):
    """
    Cargamos el .txt y lo convertimos con exec a un diccionario con el que poder trabajar.

    """
    namespace = {}
    with open(path_archivo, "r", encoding="utf-8") as f:
        exec(f.read(), {}, namespace)
    return namespace["colecciones"]
'''
def detectar_neoplasias(texto, c):
    colecciones = cargar_colecciones(c)
    texto_norm = normalizar(texto)
    neos_detectadas = []

    for neo, terminos in colecciones.items():
        for termino in terminos:
            if re.search(r"\b" + re.escape(termino) + r"\b", texto_norm):
                neos_detectadas.append(neo)
                break  

    return list(set(neos_detectadas))

    '''

def detectar_neoplasias(texto, colecciones_norm):
    texto_norm = normalizar(texto)
    neos_detectadas = []
    for neo, terminos in colecciones_norm.items():
        for termino in terminos:
            if re.search(r"\b" + re.escape(termino) + r"\b", texto_norm):
                neos_detectadas.append(neo)
                break
    return list(set(neos_detectadas))


def clasificar(df, colecciones):
    df["neoplasias_nuevas"]= df["TEXTO"].apply(lambda paciene: detectar_neoplasias(paciene, colecciones))
    df["multiples_nuevas"] = df["neoplasias_nuevas"].apply( lambda neoplasias:1 if len(neoplasias)>1 else 0)
    return df


def parse_neoplasias(valor):
    """
    Convierte la columna NEOPLASIAS (que viene como string) en una lista.
    Ej: "['Pulmón', 'Colon']" -> ['Pulmón', 'Colon']
    """
    if pd.isna(valor):
        return []
    if isinstance(valor, list):
        return valor
    if isinstance(valor, str):
        try:
            return ast.literal_eval(valor)
        except:
            return []
    return []


def evaluar(df,output_folder):
    y_true= df["MULTIPLES"]
    y_pred = df["multiples_nuevas"]
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)


    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "metricas_clasificacion.txt"), "w") as f:
        f.write(f"""AUC: {auc:.4f}
                    Accuracy: {accuracy:.4f}
                    F1: {f1:.4f}
                    Precision: {precision:.4f}
                    Recall: {recall:.4f}

                    Matriz de Confusión:
                    {cm.tolist()}
                """)

    df_fallos = df[df["MULTIPLES"] != df["multiples_nuevas"]]
    df_fallos.to_csv(os.path.join(output_folder, "fallos_clasificacion.csv"), index=False)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicción")
    plt.ylabel("Clase Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "matriz_confusion.png"))
    plt.close()

def evaluar_flexible(df, output_folder):
    """
    Evaluación flexible: se considera correcto si se acierta al menos una de las neoplasias reales.
    También guarda resumen por tipo de paciente en el mismo TXT de métricas.
    """

    df["acierto_flexible"] = df.apply(
        lambda fila: int(bool({normalizar(neo) for neo in parse_neoplasias(fila["NEOPLASIAS"])}.intersection({normalizar(neo) for neo in fila["neoplasias_nuevas"]}))),
        axis=1
    )

    

    os.makedirs(output_folder, exist_ok=True)

    
    resumen = df.groupby("MULTIPLES")["acierto_flexible"].agg(
        total="count",
        aciertos="sum",
        porcentaje = lambda x: x.sum() / x.count() * 100,
        errores=lambda x: (1 - x).sum()
    ).reset_index()

    with open(os.path.join(output_folder, "metricas_flexibles.txt"), "w") as f:
        f.write(f"""Evaluación flexible (al menos una neoplasia correcta):
           Aciertos totales: {df["acierto_flexible"].sum()} de {len(df)} , porcentaje: {df["acierto_flexible"].sum() / len(df) * 100:.2f}%
            Resumen por tipo de paciente: \n
            """)
        for _, row in resumen.iterrows():
            f.write(f"- {row['MULTIPLES']}: total={row['total']}, aciertos={row['aciertos']}, errores={row['errores']}, porcentaje={row['porcentaje']} \n")

    # Guardar fallos
    df_errores = df[df["acierto_flexible"] == 0]
    df_errores.to_csv(os.path.join(output_folder, "fallos_flexibles.csv"), index=False)


def es_acierto_estricto(fila):
    """
    Comprueba si todas las neoplasias reales (sin duplicados) están en la predicción.
    Normaliza antes de comparar (minúsculas, sin acentos).
    """
    reales = parse_neoplasias(fila["NEOPLASIAS"])
    predichas = fila.get("neoplasias_nuevas", []) or []
    
    reales_unicos = list(set(reales))
    reales_normalizados = {normalizar(neo) for neo in reales_unicos}
    predichas_normalizados = {normalizar(neo) for neo in predichas}
    
    return int(all(neo in predichas_normalizados for neo in reales_normalizados))


def evaluar_estricto(df, output_folder):
    """
    Evaluación estricta: debe acertar TODAS las neoplasias reales (sin duplicados).
    Si la neoplasia real se repite (ej: ['Mama', 'Mama']), basta detectarla una sola vez.
    
    Ejemplos:
    - Real: ['Pulmón', 'Mama'], Predicción: ['Pulmón', 'Mama'] → 1 (correcto)
    - Real: ['Pulmón', 'Mama'], Predicción: ['Pulmón'] → 0 (incorrecto, falta Mama)
    - Real: ['Mama', 'Mama'], Predicción: ['Mama'] → 1 (correcto, se repite)
    - Real: ['Mama', 'Mama'], Predicción: ['Pulmón'] → 0 (incorrecto)
    """
    
    df["acierto_estricto"] = df.apply(es_acierto_estricto, axis=1)
    
    os.makedirs(output_folder, exist_ok=True)
    
    resumen = df.groupby("MULTIPLES")["acierto_estricto"].agg(
        total="count",
        aciertos="sum",
        porcentaje=lambda x: x.sum() / x.count() * 100,
        errores=lambda x: (1 - x).sum()
    ).reset_index()
    
    with open(os.path.join(output_folder, "metricas_estrictas.txt"), "w") as f:
        f.write(f"""Evaluación estricta (debe acertar TODAS las neoplasias reales):
            Aciertos totales: {df["acierto_estricto"].sum()} de {len(df)}, porcentaje: {df["acierto_estricto"].sum() / len(df) * 100:.2f}%

            Resumen por tipo de paciente:
            """)
        for _, row in resumen.iterrows():
            f.write(f"- {row['MULTIPLES']}: total={row['total']}, aciertos={row['aciertos']}, errores={row['errores']}, porcentaje={row['porcentaje']:.2f}%\n")
    
    df_errores = df[df["acierto_estricto"] == 0]
    df_errores.to_csv(os.path.join(output_folder, "fallos_estrictos.csv"), index=False)


### MAIN

coleccion = r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\coleccion\coleccion.txt"
colecciones_normalizadas = cargar_colecciones2(coleccion)
df = pd.read_csv(r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\comparacionModelos\datos\test_set_completo_cambiado.csv", sep=';')
output_folder= r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\prediccion_coleccion\prediccion2"

df_clasificado = clasificar(df,colecciones_normalizadas)
evaluar(df_clasificado, output_folder)
evaluar_flexible(df_clasificado, output_folder)
evaluar_estricto(df_clasificado, output_folder)