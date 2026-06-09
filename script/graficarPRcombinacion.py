import pandas as pd
import matplotlib.pyplot as plt
import os

ruta_csv = r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\comparacionModelos\comparacionTestCompletos\metricas_por_combinacion_TODOS_LOS_MODELOS.csv"

salida_dir = os.path.join(os.path.dirname(ruta_csv), "graficas_f1_recall")
os.makedirs(salida_dir, exist_ok=True)

df = pd.read_csv(ruta_csv)

combinaciones = df["Combinación"].unique()

for combinacion in combinaciones:
    df_sub = df[df["Combinación"] == combinacion]

    modelos = df_sub["Modelo"].tolist()
    f1 = df_sub["F1"].tolist()
    recall = df_sub["Recall"].tolist()
    soporte = df_sub["Soporte"].tolist()

    x = range(len(modelos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar([i - width/2 for i in x], f1, width, label='F1', color='tab:blue')
    bars2 = ax.bar([i + width/2 for i in x], recall, width, label='Recall', color='tab:orange')

    for i, s in enumerate(soporte):
        ax.text(i, max(f1[i], recall[i]) + 0.02, f"Soporte: {s}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([i/10 for i in range(0, 11)])  
    ax.set_ylabel("Valor")
    ax.set_title(f"F1 y Recall - {combinacion}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    nombre_archivo = combinacion.replace(",", "_").replace(" ", "_").replace("/", "_") + ".png"
    plt.tight_layout()
    plt.savefig(os.path.join(salida_dir, nombre_archivo), dpi=300)
    plt.close()

print(f"Gráficas guardadas en: {salida_dir}")
