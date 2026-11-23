import os
import csv
import numpy as np
from typing import List


def init_summary_csv(out_dir: str, filename: str = "resumen_energias_alpha2.csv") -> str:
    #resumen de energias para alpha = 2
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["L", "alpha", "h", "E1", "E2", "E3", "is_reference", "wave_samples_csv"])
    return path


def append_summary_row(summary_path: str, L: float, alpha: float, h: float,
                       E1: float, E2: float, E3: float, is_reference: int,
                       wave_samples_csv: str):
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([L, alpha, h, E1, E2, E3, is_reference, wave_samples_csv])


def init_wave_samples_csv_for_L(out_dir: str, L: float, alpha: float, x_points: List[float]) -> str:
    #csv de autofunciones
    os.makedirs(out_dir, exist_ok=True)
    filename = f"wave_samples_L{L:g}_alpha{alpha:g}.csv"
    path = os.path.join(out_dir, filename)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["L", "alpha", "h", "is_reference", "n", "E"] + [f"phi(x={x:+.3f})" for x in x_points]
        w.writerow(header)

    return path


def append_wave_samples_row(wave_samples_path: str, L: float, alpha: float, h: float,
                            is_reference: int, n: int, E: float, phi_samples: List[float]):
    with open(wave_samples_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([L, alpha, h, is_reference, n, E] + list(phi_samples))



def normalize_phi(xs: np.ndarray, phi: np.ndarray) -> np.ndarray:
    #normaliza phi con metodo trapezoidal
    norm2 = np.trapezoid(phi * phi, xs)
    if not np.isfinite(norm2) or norm2 <= 0:
        return phi
    return phi / np.sqrt(norm2)


def sample_phi_at_points(xs: np.ndarray, phi: np.ndarray, x_points: List[float], tol: float = 1e-12) -> List[float]:
    #hace que los puntos de la malla sean iguales para diferentes tamaños
    dx = xs[1] - xs[0]
    x0 = xs[0]
    out = []

    for x in x_points:
        idx = int(round((x - x0) / dx))
        if idx < 0 or idx >= len(xs):
            raise ValueError(f"x={x} está fuera de la malla [{xs[0]}, {xs[-1]}].")
        if not np.isclose(xs[idx], x, atol=tol, rtol=0):
            raise ValueError(
                f"x={x} no cae exactamente en la malla: xs[{idx}]={xs[idx]} (dx={dx}). "
                f"Elige x_points compatibles con tus h."
            )
        out.append(float(phi[idx]))

    return out
