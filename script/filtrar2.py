

import pandas as pd
from pathlib import Path

# Rutas
ruta_original = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/Multiples_neoplasias_solo_resumenes_selection/Multiples_neoplasias_solo_resumenes_deidentified_selection.csv"  # contiene todas las columnas
ruta_splits = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/NeoplasiaClassifier-ES/output/comparacionModelos/datos"  # carpeta donde están los train/val/test actuales


df_full = pd.read_csv(ruta_original, sep=";", encoding="utf-8", quotechar='"')
df_full["TEXTO"] = df_full["TEXTO"].astype(str)
df_full["MULTIPLES"] = df_full["MULTIPLES"].astype(int)

# Crear una función auxiliar para generar claves por n palabras
def generar_claves(df, max_n=20):
    claves = {}
    for idx, row in df.iterrows():
        texto = str(row["TEXTO"])
        for n in range(7, max_n + 1):
            clave = " ".join(texto.split()[:n])
            if clave not in claves:
                claves[clave] = row
    return claves

# Crear diccionario de claves desde el dataset completo
claves_full = generar_claves(df_full, max_n=30)

# Procesar cada split sin modificar el original
for nombre in ["train", "val", "test"]:
    path_split = Path(ruta_splits) / f"{nombre}_set.csv"
    df_split = pd.read_csv(path_split)
    df_split["TEXTO"] = df_split["TEXTO"].astype(str)
    df_split["MULTIPLES"] = df_split["MULTIPLES"].astype(int)

    datos_enriquecidos = []

    for idx, row in df_split.iterrows():
        texto_split = row["TEXTO"]
        match = None
        for n in range(7, 31):
            clave = " ".join(texto_split.split()[:n])
            if clave in claves_full:
                match = claves_full[clave]
                break
        if match is not None:
            fila_completa = row.to_dict()
            for col in df_full.columns:
                if col not in fila_completa:
                    fila_completa[col] = match[col]
            datos_enriquecidos.append(fila_completa)
        else:
            print(f"No se encontró coincidencia para índice {idx} en {nombre} con texto: {texto_split[:60]}...")

        # Crear DataFrame final con filas enriquecidas
    df_completo = pd.DataFrame(datos_enriquecidos)

    # Reordenar columnas: ID primero, TIPO_NEO al final si existen
    columnas = list(df_completo.columns)
    if "ID_PACIENTE" in columnas:
        columnas_reordenadas = ["ID_PACIENTE"] + [col for col in columnas if col != "ID_PACIENTE"]
        df_completo = df_completo[columnas_reordenadas]
    else:
        print(f"[!] Aviso: columna 'ID' no encontrada en {nombre}, no se reordenó.")

    # Guardar CSV con separador ;
    df_completo.to_csv(
        Path(ruta_splits) / f"{nombre}_set_completo.csv",
        index=False,
        sep=";",              # <-- separador correcto
        encoding="utf-8",
        quotechar='"'
    )

    print(f"{nombre.capitalize()} enriquecido y guardado con {df_completo.shape[0]} filas.")
