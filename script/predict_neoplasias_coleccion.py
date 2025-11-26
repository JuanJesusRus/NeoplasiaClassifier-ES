import pandas as pd
import re
import unicodedata
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
    # Normalizar todos los términos en la colección
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

### MAIN

coleccion = r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\coleccion\coleccion.txt"
colecciones_normalizadas = cargar_colecciones2(coleccion)
df = pd.read_csv(r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\comparacionModelos\datos\test_set_completo_cambiado.csv", sep=';')
output_folder= r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\prediccion_coleccion\prediccion2"


evaluar(clasificar(df,colecciones_normalizadas), output_folder)