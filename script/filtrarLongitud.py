from transformers import AutoTokenizer
import pandas as pd



INPUT_PATH = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/Multiples_neoplasias_solo_resumenes_selection/Multiples_neoplasias_solo_resumenes_deidentified_selection.csv"

df = pd.read_csv(INPUT_PATH, sep=";", encoding="utf-8", quotechar='"')


modelo_path = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
tokenizer = AutoTokenizer.from_pretrained(modelo_path)

# Volvemos a recorrer el dataset para separar los largos también
textos_cortos, clases_cortas = [], []
textos_largos, clases_largos = [], []

for texto, clase in zip(df["TEXTO"], df["MULTIPLES"]):
    tokenizado = tokenizer.encode(texto, truncation=False, add_special_tokens=True)
    if len(tokenizado) <= 512:
        textos_cortos.append(texto)
        clases_cortas.append(clase)
    else:
        textos_largos.append(texto)
        clases_largos.append(clase)

# Creamos los DataFrames
df_cortos = pd.DataFrame({"TEXTO": textos_cortos, "MULTIPLES": clases_cortas})
df_largos = pd.DataFrame({"TEXTO": textos_largos, "MULTIPLES": clases_largos})


ruta_guardado = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/basura/bas"

# Guardar los DataFrames
df_cortos.to_csv(f"{ruta_guardado}/textos_cortos_filtrados.csv", index=False)
df_largos.to_csv(f"{ruta_guardado}/textos_largos_filtrados.csv", index=False)

print (df_cortos)
print(df_largos)