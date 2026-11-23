import os
import csv
import numpy as np

from potencial import energia_estado_n


def energias_oscilador_libre(alpha: float):
    #Energias os armon libre
    if alpha <= 0.0:
        return [np.nan, np.nan, np.nan]  # no tiene sentido HO si alpha=0
    # Free levels of -phi'' + alpha*x^2*phi = E*phi are E_n = (n - 1/2) * 2*sqrt(alpha)
    omega = 2.0 * np.sqrt(alpha)
    return [0.5 * omega, 1.5 * omega, 2.5 * omega]  # n=1,2,3


def main():
    # ============================================================
    # Apartado (b): barrido en L, alpha, h
    # ============================================================
    L_values = [1.0, 2.0, 3.0]

    # varios alpha
    alpha_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Varios mallados
    h_values = [0.05, 0.02, 0.01, 0.005]

    out_dir = "resultados_apartado_b"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "resumen_energias_L_alpha_h_con_HO.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow([
            "L", "alpha", "h",
            "E1", "E2", "E3",
            "E1_HO", "E2_HO", "E3_HO",
            "DeltaE1", "DeltaE2", "DeltaE3"
        ])

        for L in L_values:
            print(f"\n=== L={L} ===")

            for alpha in alpha_values:
                # Energías osc (sin paredes) para comparar
                EHO1, EHO2, EHO3 = energias_oscilador_libre(alpha)

                # Seeds iniciales: si alpha=0 usa pozo plano exacto; si alpha>0 usa HO
                if alpha == 0.0:
                    guess = [
                        (1 ** 2) * (np.pi ** 2) / (8.0 * L ** 2),
                        (2 ** 2) * (np.pi ** 2) / (8.0 * L ** 2),
                        (3 ** 2) * (np.pi ** 2) / (8.0 * L ** 2),
                    ]
                else:
                    guess = [EHO1, EHO2, EHO3]

                for h in h_values:
                    deltas = [1.0, 1.0, 1.0]

                    E1 = energia_estado_n(1, L, alpha, h, eps_inicial=guess[0], delta_eps_inicial=deltas[0])
                    E2 = energia_estado_n(2, L, alpha, h, eps_inicial=max(E1, guess[1]), delta_eps_inicial=deltas[1])
                    E3 = energia_estado_n(3, L, alpha, h, eps_inicial=max(E2, guess[2]), delta_eps_inicial=deltas[2])

                    # actualiza seeds para el siguiente h
                    guess = [E1, E2, E3]

                    dE1 = E1 - EHO1 if np.isfinite(EHO1) else np.nan
                    dE2 = E2 - EHO2 if np.isfinite(EHO2) else np.nan
                    dE3 = E3 - EHO3 if np.isfinite(EHO3) else np.nan

                    w.writerow([L, alpha, h, E1, E2, E3, EHO1, EHO2, EHO3, dE1, dE2, dE3])

                    print(f"L={L} alpha={alpha:5.2f} h={h:7.4f} | "
                          f"E1={E1:11.6f} E2={E2:11.6f} E3={E3:11.6f} | "
                          f"ΔE1={dE1:10.6f} ΔE2={dE2:10.6f} ΔE3={dE3:10.6f}")

    print(f"\nCSV guardado en: {out_csv}")


if __name__ == "__main__":
    main()
