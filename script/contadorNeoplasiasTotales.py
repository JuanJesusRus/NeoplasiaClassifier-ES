import ast
from collections import Counter

ruta_csv = r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\Multiples_neoplasias_solo_resumenes_selection\Multiples_neoplasias_solo_resumenes_deidentified_selection.csv"
ruta_salida = r"C:\Users\jesus\OneDrive - Universidad de Málaga\Cuarto\TFG\NeoplasiaClassifier-ES\output\eda\combinaciones_neoplasias2.txt"
encoding = "utf-8-sig"
'''
def try_parse_list(s):
    """Devuelve lista si s es una lista estilo Python/JSON, o [] si no."""
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val]
    except Exception:
        pass
    return []

conteo = Counter()
total = 0
malformadas = 0

with open(ruta_csv, "r", encoding=encoding) as f:
    header = f.readline()  # saltamos cabecera
    for i, line in enumerate(f, start=2):
        line = line.rstrip("\n")
        if not line:
            continue

        # Divide por las últimas 3 apariciones de ";"
        parts = line.rsplit(";", 3)
        if len(parts) != 4:
            malformadas += 1
            continue

        id_paciente, texto, col3, col4 = parts

        # Detectamos automáticamente qué columna es NEOPLASIAS
        neos = try_parse_list(col3)
        if not neos:
            neos = try_parse_list(col4)

        if neos:
            combo = tuple(sorted(neos))
            conteo[combo] += 1
            total += 1

# === MOSTRAR RESULTADOS EN PANTALLA ===
print("=== Combinaciones de neoplasias y frecuencia ===")
for combo, count in sorted(conteo.items(), key=lambda x: (-x[1], x[0])):
    print(f"{combo}: {count}")

print("\nResumen:")
print(f"Filas con NEOPLASIAS válidas contadas: {total}")
print(f"Filas mal formadas (no 4 partes con rsplit(';', 3)): {malformadas}")

# === GUARDAR RESULTADOS EN TXT ===
with open(ruta_salida, "w", encoding="utf-8") as out:
    out.write("=== Combinaciones de neoplasias y frecuencia ===\n")
    for combo, count in sorted(conteo.items(), key=lambda x: (-x[1], x[0])):
        out.write(f"{combo}: {count}\n")
    out.write("\nResumen:\n")
    out.write(f"Filas con NEOPLASIAS válidas contadas: {total}\n")
    out.write(f"Filas mal formadas: {malformadas}\n")

print(f"\n✅ Resultados guardados en: {ruta_salida}")

'''


def try_parse_list(s):
    """Devuelve lista si s es una lista estilo Python/JSON, o [] si no."""
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val]
    except Exception:
        pass
    return []

conteo_combinaciones = Counter()
conteo_individual = Counter()
total = 0
malformadas = 0

with open(ruta_csv, "r", encoding=encoding) as f:
    header = f.readline()  # saltamos cabecera
    for i, line in enumerate(f, start=2):
        line = line.rstrip("\n")
        if not line:
            continue

        # Divide por las últimas 3 apariciones de ";"
        parts = line.rsplit(";", 3)
        if len(parts) != 4:
            malformadas += 1
            continue

        id_paciente, texto, col3, col4 = parts

        # Detectamos automáticamente qué columna es NEOPLASIAS
        neos = try_parse_list(col3)
        if not neos:
            neos = try_parse_list(col4)

        if neos:
            combo = tuple(sorted(neos))
            conteo_combinaciones[combo] += 1
            for n in neos:
                conteo_individual[n] += 1
            total += 1

# === MOSTRAR RESULTADOS EN PANTALLA ===
print("=== Combinaciones de neoplasias y frecuencia ===")
for combo, count in sorted(conteo_combinaciones.items(), key=lambda x: (-x[1], x[0])):
    print(f"{combo}: {count}")

print("\n=== Frecuencia individual de cada tipo de neoplasia ===")
for n, count in sorted(conteo_individual.items(), key=lambda x: (-x[1], x[0])):
    print(f"{n}: {count}")

print("\nResumen:")
print(f"Filas con NEOPLASIAS válidas contadas: {total}")
print(f"Filas mal formadas (no 4 partes con rsplit(';', 3)): {malformadas}")

# === GUARDAR RESULTADOS EN TXT ===
with open(ruta_salida, "w", encoding="utf-8") as out:
    out.write("=== Combinaciones de neoplasias y frecuencia ===\n")
    for combo, count in sorted(conteo_combinaciones.items(), key=lambda x: (-x[1], x[0])):
        out.write(f"{combo}: {count}\n")

    out.write("\n=== Frecuencia individual de cada tipo de neoplasia ===\n")
    for n, count in sorted(conteo_individual.items(), key=lambda x: (-x[1], x[0])):
        out.write(f"{n}: {count}\n")

    out.write("\nResumen:\n")
    out.write(f"Filas con NEOPLASIAS válidas contadas: {total}\n")
    out.write(f"Filas mal formadas: {malformadas}\n")
