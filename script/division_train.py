import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# Configuración
config = {
    "ruta_csv": r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\Multiples_neoplasias_solo_resumenes_selection\textos_cortos_filtrados.csv",  # <-- cámbialo por la ruta a tu CSV
    "columna_texto": "TEXTO",  # <-- cámbialo
    "columna_etiqueta": "MULTIPLES",  # <-- cámbialo
    "test_size": 0.2,
    "random_state": 42,
    "ruta_salida": r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\comparacionModelos\datos"  # <-- carpeta donde guardar los CSVs
}

# Crear carpeta de salida si no existe
Path(config["ruta_salida"]).mkdir(parents=True, exist_ok=True)

# Rutas completas de los archivos de salida
output_files = {
    "train": os.path.join(config["ruta_salida"], "train2_set.csv"),
    "val": os.path.join(config["ruta_salida"], "val2_set.csv"),
    "test": os.path.join(config["ruta_salida"], "test2_set.csv")
}

# Solo generamos los archivos si no existen
# Solo generar los splits si no existen ya
if all(not Path(f).exists() for f in output_files.values()):
    # Leer el CSV tal como está
    df = pd.read_csv(config["ruta_csv"])

    # División train / temp (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=config["test_size"],
        stratify=df[config["columna_etiqueta"]],
        random_state=config["random_state"]
    )

    # División val / test
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp[config["columna_etiqueta"]],
        random_state=config["random_state"]
    )

    # Guardar los resultados
    df_train.to_csv(output_files["train"], index=False)
    df_val.to_csv(output_files["val"], index=False)
    df_test.to_csv(output_files["test"], index=False)

    print("Conjuntos train, val y test guardados en:", config["ruta_salida"])
else:
    print("Los archivos ya existen. No se han sobrescrito.")