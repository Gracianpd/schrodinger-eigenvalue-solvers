import os
import glob
import re
import csv
import numpy as np
import matplotlib.pyplot as plt


def parse_x_from_header(colname: str) -> float:
    """
    Extrae x de strings tipo:
      'phi(x=-0.900)'  o  'phi(x=+1.300)'
    """
    m = re.search(r"phi\(x=([+-]?\d+(?:\.\d+)?)\)", colname.strip())
    if not m:
        raise ValueError(f"No puedo extraer x de la columna: {colname}")
    return float(m.group(1))


def read_wave_samples_csv(path: str):
    """
    Lee el CSV wave_samples_L*_alpha*.csv
    Devuelve: header(list), rows(list of dict)
    """
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


def plot_reference_points_for_file(path: str, out_dir: str):
    header, rows = read_wave_samples_csv(path)

    # columnas esperadas:
    # ["L","alpha","h","is_reference","n","E", "phi(x=...)", ...]
    needed = {"L", "alpha", "h", "is_reference", "n", "E"}
    if not needed.issubset(set(header)):
        raise ValueError(f"Cabecera inesperada en {path}. Columnas: {header}")

    # extraer x_points de las columnas phi(x=...)
    phi_cols = [c for c in header if c.startswith("phi(x=")]
    x_points = np.array([parse_x_from_header(c) for c in phi_cols], dtype=float)

    # filtrar referencia
    ref_rows = [r for r in rows if str(r["is_reference"]).strip() == "1"]
    if not ref_rows:
        raise RuntimeError(f"No hay filas de referencia (is_reference=1) en {path}.")

    # L y alpha para título
    L_val = float(ref_rows[0]["L"])
    alpha_val = float(ref_rows[0]["alpha"])
    h_ref = float(ref_rows[0]["h"])

    # ordenar puntos por x (por si en el CSV están en otro orden)
    order = np.argsort(x_points)
    x_sorted = x_points[order]

    plt.figure()
    for n in (1, 2, 3):
        # coge la fila de referencia de ese n
        candidates = [r for r in ref_rows if int(float(r["n"])) == n]
        if not candidates:
            print(f"  Aviso: en {os.path.basename(path)} falta referencia para n={n}")
            continue

        r = candidates[0]
        phi_vals = np.array([float(r[c]) for c in phi_cols], dtype=float)[order]

        # puntos (y opcional: unir con línea suave para guiar la vista)
        plt.plot(x_sorted, phi_vals, marker="o", linestyle="-", label=f"n={n}")

    plt.xlabel("x")
    plt.ylabel("phi(x)")
    plt.title(f"Autofunciones  L={L_val:g}, alpha={alpha_val:g}, h_ref={h_ref:g}")
    plt.grid(True)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phi_ref_puntos_L{L_val:g}_alpha{alpha_val:g}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Guardado: {out_path}")


def main():
    # Cambia esto si tu carpeta de resultados se llama distinto
    data_dir = "resultados_pozo_parabolico"
    out_dir = os.path.join(data_dir, "figuras_autofunciones_ref")

    files = sorted(glob.glob(os.path.join(data_dir, "wave_samples_L*_alpha*.csv")))
    if not files:
        raise RuntimeError(f"No encuentro wave_samples_L*_alpha*.csv en {data_dir}")

    for path in files:
        plot_reference_points_for_file(path, out_dir)


if __name__ == "__main__":
    main()