import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIGURACIÓN ===
INPUT_PATH = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/Multiples_neoplasias_solo_resumenes_selection/Multiples_neoplasias_solo_resumenes_deidentified_selection.csv"     # Ruta al archivo CSV (asegúrate de que exista)
TXT_OUTPUT = "output/resumen_datos.txt"         # Archivo resumen de salida
IMG_FOLDER = "output/img"                       # Carpeta de imágenes

# === CREAR CARPETAS SI NO EXISTEN ===
os.makedirs(IMG_FOLDER, exist_ok=True)

# === CARGA DE DATOS DESDE CSV ===
try:
    df = pd.read_csv(INPUT_PATH, sep=";", encoding="utf-8", quotechar='"')

except FileNotFoundError:
    print(f"❌ Archivo no encontrado: {INPUT_PATH}")
    exit()
except Exception as e:
    print(f"❌ Error al leer el archivo: {e}")
    exit()

# Mostrar columnas detectadas
print("🧾 Columnas detectadas:", df.columns.tolist())

# Detectar columnas automáticamente
col_texto = df.columns[1]   # Historial clínico
col_clase = df.columns[-1]  # Clase binaria (0 o 1)

# === CALCULAR NÚMERO DE PALABRAS ===
df["Num_Palabras"] = df[col_texto].apply(lambda x: len(str(x).split()))

# === ESTADÍSTICAS BÁSICAS ===
conteo_clases = df[col_clase].value_counts().sort_index()
rango_palabras = df.groupby(col_clase)["Num_Palabras"].agg(["min", "max", "mean", "std"])

# === GUARDAR RESUMEN EN TXT ===
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("📊 Conteo por clase (0 = una neoplasia, 1 = múltiples neoplasias):\n")
    f.write(str(conteo_clases) + "\n\n")
    f.write("📈 Rango de palabras por clase:\n")
    f.write(str(rango_palabras.round(2)))

print(f"✅ Resumen guardado en {TXT_OUTPUT}")

# === GRÁFICO 1: DISTRIBUCIÓN GENERAL ===
plt.figure(figsize=(10, 6))
plt.hist(df["Num_Palabras"], bins=30, color="skyblue", edgecolor="black")
plt.title("Distribución del número de palabras en los historiales")
plt.xlabel("Número de palabras")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{IMG_FOLDER}/distribucion_palabras_total.png")
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
plt.savefig(f"{IMG_FOLDER}/distribucion_palabras_por_clase.png")
plt.close()

print(f"📊 Gráficos guardados en {IMG_FOLDER}")