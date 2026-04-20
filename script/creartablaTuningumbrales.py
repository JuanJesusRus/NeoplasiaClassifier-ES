import json
from pathlib import Path
import pandas as pd
import argparse
import sys


def flatten_record(record: dict, experiment_name: str) -> dict:
    row = {
        "experiment": experiment_name,
        "type": record.get("type"),
    }

    # thresholds
    thresholds = record.get("thresholds", {})
    for k, v in thresholds.items():
        row[k] = v

    # cutoff
    if "support_cutoff" in record:
        row["support_cutoff"] = record["support_cutoff"]

    # metrics
    metrics = record.get("metrics", {})
    for k, v in metrics.items():
        row[k] = v

    return row


def collect_all(base_dirs: list):
    rows = []
    processed_files = set()  # Rastrear archivos ya procesados

    for base_dir in base_dirs:
        base_path = Path("output") / base_dir
        json_files = list(base_path.rglob("metrics_by_threshold.json"))

        for jf in json_files:
            abs_path = jf.resolve()
            if abs_path in processed_files:
                print(f"Duplicado omitido: {jf}")
                continue
            processed_files.add(abs_path)

            exp_name = f"{base_dir}_{jf.parent.name}" 

            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            for record in data:
                row = flatten_record(record, exp_name)

                if "modelA" in exp_name:
                    row["modelo"] = "A"
                elif "modelB" in exp_name:
                    row["modelo"] = "B"

                if "cutoff5" in exp_name:
                    row["cutoff"] = 5
                elif "cutoff10" in exp_name:
                    row["cutoff"] = 10
                elif "cutoff15" in exp_name:
                    row["cutoff"] = 15

                rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_best_per_experiment(df: pd.DataFrame, base_dirs: list):
    """
    Selecciona la mejor fila por experimento leyendo de best_threshold.json
    """
    best_rows = []

    for base_dir in base_dirs:
        base_path = Path("output") / base_dir
        for subdir in base_path.iterdir():
            if subdir.is_dir() and (subdir / "best_threshold.json").exists():
                exp_name = f"{base_dir}_{subdir.name}"

                # Leer best_threshold.json
                with open(subdir / "best_threshold.json", "r", encoding="utf-8") as f:
                    best_data = json.load(f)

                best_thresholds = best_data.get("best_thresholds", {})
                best_metrics = best_data.get("best_metrics", {})

                # Filtrar df por experimento
                sub = df[df["experiment"] == exp_name]

                # Buscar la fila que coincida con best_thresholds
                matching_row = None
                for _, row in sub.iterrows():
                    match = True
                    for k, v in best_thresholds.items():
                        if row.get(k) != v:
                            match = False
                            break
                    if match:
                        matching_row = row
                        break

                if matching_row is not None:
                    best_rows.append(matching_row)

    return pd.DataFrame(best_rows)


def flatten_test_metrics(record: dict, experiment_name: str) -> dict:
    row = {
        "experiment": experiment_name,
    }

    # Add all metrics from the record
    for k, v in record.items():
        row[k] = v

    return row


def collect_test_metrics(base_dirs: list):
    rows = []
    processed_files = set()  # Rastrear archivos ya procesados para evitar duplicados

    for base_dir in base_dirs:
        base_path = Path("output") / base_dir
        json_files = list(base_path.rglob("test_metrics.json"))

        for jf in json_files:
            abs_path = jf.resolve()
            if abs_path in processed_files:
                print(f"Duplicado omitido: {jf}")
                continue
            processed_files.add(abs_path)

            exp_name = f"{base_dir}_{jf.parent.name}"

            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            row = flatten_test_metrics(data, exp_name)

            # Load best_thresholds from best_threshold.json if exists
            best_threshold_path = jf.parent / "best_threshold.json"
            if best_threshold_path.exists():
                with open(best_threshold_path, "r", encoding="utf-8") as f:
                    best_data = json.load(f)
                best_thresholds = best_data.get("best_thresholds", {})
                for k, v in best_thresholds.items():
                    row[k] = v

            # Determine model and cutoff from experiment name
            if "modelA" in exp_name:
                row["modelo"] = "A"
            elif "modelB" in exp_name:
                row["modelo"] = "B"

            if "cutoff5" in exp_name:
                row["cutoff"] = 5
            elif "cutoff10" in exp_name:
                row["cutoff"] = 10
            elif "cutoff15" in exp_name:
                row["cutoff"] = 15

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Crear tablas de métricas desde múltiples carpetas base en output/.")
    parser.add_argument("base_dirs", nargs="+", help="Nombres de carpetas base dentro de output/ donde buscar archivos JSON.")
    parser.add_argument("--mode", choices=["thresholds", "test"], default="thresholds", help="Modo: 'thresholds' para metrics_by_threshold.json, 'test_metrics' para test_metrics.json.")
    parser.add_argument("--output_dir", default="output", help="Directorio de salida para las tablas.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.mode == "thresholds":
        df = collect_all(args.base_dirs)

        # Guardar tabla completa
        df.to_csv(output_dir / "tabla_completa_thresholds.csv", index=False, encoding="utf-8-sig")
        df.to_excel(output_dir / "tabla_completa_thresholds.xlsx", index=False)

        # Filtrar ruido (opcional)
        df_filtrado = df[df["pred_labels_p90"] <= 3]

        # Mejor por experimento
        best_df = get_best_per_experiment(df_filtrado, args.base_dirs)

        best_df.to_csv(output_dir / "mejores_por_experimento.csv", index=False, encoding="utf-8-sig")
        best_df.to_excel(output_dir / "mejores_por_experimento.xlsx", index=False)

        print(f"Tablas de thresholds guardadas en {output_dir}")

    elif args.mode == "test":
        df = collect_test_metrics(args.base_dirs)

        # Guardar tabla completa
        df.to_csv(output_dir / "tabla_test_metrics_completa.csv", index=False, encoding="utf-8-sig")
        df.to_excel(output_dir / "tabla_test_metrics_completa.xlsx", index=False)

        print(f"Tabla de test_metrics guardada en {output_dir / 'tabla_test_metrics_completa.csv'} y {output_dir / 'tabla_test_metrics_completa.xlsx'}")


if __name__ == "__main__":
    main()