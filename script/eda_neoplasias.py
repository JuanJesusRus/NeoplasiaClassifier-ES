import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from wordcloud import WordCloud


# === CONFIGURACIÓN ===
INPUT_PATH = r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\basura\bas\textos_cortos_filtrados.csv"   # Ruta al archivo CSV (asegúrate de que exista)  
TXT_OUTPUT = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/NeoplasiaClassifier-ES/output/eda/resumenDatosfiltrado.txt"       # Archivo resumen de salida
IMG_FOLDER = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/NeoplasiaClassifier-ES/output/eda"                       # Carpeta de imágenes

# === CREAR CARPETAS SI NO EXISTEN ===
os.makedirs(IMG_FOLDER, exist_ok=True)

# === CARGA DE DATOS DESDE CSV ===
try:
    df = pd.read_csv(INPUT_PATH, sep=";", encoding="utf-8", on_bad_lines='skip')

except FileNotFoundError:
    print(f"Archivo no encontrado: {INPUT_PATH}")
    exit()
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    exit()

# Mostrar columnas detectadas
print("🧾 Columnas detectadas:", df.columns.tolist())

# Detectar columnas automáticamente
col_texto = "TEXTO"   # Texto clínico
col_clase = "MULTIPLES"   # Clase binaria (0 o 1)

# === CALCULAR NÚMERO DE PALABRAS ===
df["Num_Palabras"] = df[col_texto].apply(lambda x: len(str(x).split()))

# === ESTADÍSTICAS BÁSICAS ===
conteo_clases = df[col_clase].value_counts().sort_index()
rango_palabras = df.groupby(col_clase)["Num_Palabras"].agg(["min", "max", "mean", "std"])

# === GUARDAR RESUMEN EN TXT ===
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("Conteo por clase (0 = una neoplasia, 1 = múltiples neoplasias):\n")
    f.write(str(conteo_clases) + "\n\n")
    f.write("Rango de palabras por clase:\n")
    f.write(str(rango_palabras.round(2)))

print(f"Resumen guardado en {TXT_OUTPUT}")

# === GRÁFICO 1: DISTRIBUCIÓN GENERAL ===
plt.figure(figsize=(10, 6))
plt.hist(df["Num_Palabras"], bins=30, color="skyblue", edgecolor="black", density=True)
plt.title("Distribución del número de palabras en los historiales")
plt.xlabel("Número de palabras")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{IMG_FOLDER}/distribucion_palabras_total_filtrado.png")
plt.close()

# === GRÁFICO 2: DISTRIBUCIÓN POR CLASE ===
plt.figure(figsize=(10, 6))
for clase in sorted(df[col_clase].unique()):
    subset = df[df[col_clase] == clase]
    plt.hist(subset["Num_Palabras"], bins=30, alpha=0.5, label=f"Clase {clase}")
plt.title("Distribución de palabras por clase (0 = una, 1 = múltiples neoplasias)")
plt.xlabel("Número de palabras")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{IMG_FOLDER}/distribucion_palabras_por_clase_filtrado.png")
plt.close()

print(f"Gráficos guardados en {IMG_FOLDER}")

sns.violinplot(x=col_clase, y="Num_Palabras", data=df, palette=["skyblue", "salmon"], cut=0)
plt.title("Distribución de número de palabras por clase (Violin plot)")
plt.savefig(f"{IMG_FOLDER}/graficoviolin_filtrado.png")
plt.close()


for clase in [0, 1]:
    if clase == 0:
        nombre="Una Neoplasia"
    else:
        nombre="Múltiples Neoplasias"

    texto = " ".join(df[df[col_clase] == clase][col_texto].astype(str).values)
    wc = WordCloud(width=800, height=400, background_color="white").generate(texto)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Nube de palabras – Clase {clase}")
    plt.tight_layout()
    plt.savefig(f"{IMG_FOLDER}/nubepalabras_{nombre}_filtrado.png")
    plt.close()


conteo_clases.plot(kind='bar', color=["skyblue", "salmon"])
plt.title("Historiales Clínicos")
plt.xticks([0, 1], ["Una Neoplasia", "Múltiples Neoplasias"], rotation=0)
plt.ylabel("Historiales Clínicos")
plt.grid(axis='y')
plt.savefig(f"{IMG_FOLDER}/conteo_clases_filtrado.png")
plt.close()