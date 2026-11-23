import os
import math
import numpy as np
import matplotlib.pyplot as plt

from potencial import energia_estado_n, integra_phi


# ----------------------------
# Pozo infinito exacto en [-L, L]
# ----------------------------
def well_energy_exact(n: int, L: float) -> float:
    k = n * math.pi / (2.0 * L)
    return k * k


def well_wave_exact(n: int, x: np.ndarray, L: float) -> np.ndarray:
    # Normalizada en [-L, L]
    k = n * math.pi / (2.0 * L)
    return math.sqrt(1.0 / L) * np.sin(k * (x + L))


def l2_normalize(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    area = np.trapezoid(f * f, x)   # <- en lugar de np.trapz
    if area <= 0:
        return f
    return f / np.sqrt(area)



def align_sign(f_num: np.ndarray, f_ref: np.ndarray) -> np.ndarray:
    return -f_num if np.dot(f_num, f_ref) < 0 else f_num


def main():
    # =========================
    # Parámetros apartado (c)
    # =========================
    L = 3.0
    h = 0.005

    # alphas muy bajos (casi pozo plano)
    alpha_values = [1e-3, 5e-3, 1e-2, 1e-1]

    # Para las figuras: elige unos pocos alphas representativos
    alpha_plot = [1e-3, 5e-3, 1e-2, 1e-1]

    out_dir = "apartado_c_L3_pozo_infinito"
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # 1) CSV con energías
    # =========================
    csv_path = os.path.join(out_dir, "energias_L3_vs_pozo_infinito.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("L,h,alpha,"
                "E1_num,E1_well,"
                "E2_num,E2_well,"
                "E3_num,E3_well\n")

        # Energías exactas del pozo
        Ewell = [well_energy_exact(n, L) for n in (1, 2, 3)]

        for alpha in alpha_values:
            # buenas seed
            guess = Ewell[:]
            deltas = [0.5, 0.5, 0.5]

            Enum = []
            for n in (1, 2, 3):
                En = energia_estado_n(
                    n, L, alpha, h,
                    eps_inicial=guess[n - 1],
                    delta_eps_inicial=deltas[n - 1]
                )
                Enum.append(En)

            f.write(f"{L},{h},{alpha},"
                    f"{Enum[0]},{Ewell[0]},"
                    f"{Enum[1]},{Ewell[1]},"
                    f"{Enum[2]},{Ewell[2]}\n")

            print(f"alpha={alpha:9.2e}  "
                  f"E1={Enum[0]:.6f} (well={Ewell[0]:.6f})  "
                  f"E2={Enum[1]:.6f} (well={Ewell[1]:.6f})  "
                  f"E3={Enum[2]:.6f} (well={Ewell[2]:.6f})")

    print(f"\nCSV guardado en: {csv_path}")

    # =========================
    # 2) Figuras: phi_num vs phi_pozo_infinito
    # =========================
    for alpha in alpha_plot:
        # energías numéricas para ese alpha
        Ewell = [well_energy_exact(n, L) for n in (1, 2, 3)]
        guess = Ewell[:]

        Enum = []
        for n in (1, 2, 3):
            En = energia_estado_n(
                n, L, alpha, h,
                eps_inicial=guess[n - 1],
                delta_eps_inicial=0.5
            )
            Enum.append(En)

        # integrar numéricas (malla común)
        x, phi1 = integra_phi(Enum[0], L, alpha, h)
        x2, phi2 = integra_phi(Enum[1], L, alpha, h)
        x3, phi3 = integra_phi(Enum[2], L, alpha, h)

        if len(x) != len(x2) or np.max(np.abs(x - x2)) > 1e-12:
            raise RuntimeError("Mallas distintas entre n=1 y n=2. Revisa integra_phi.")
        if len(x) != len(x3) or np.max(np.abs(x - x3)) > 1e-12:
            raise RuntimeError("Mallas distintas entre n=1 y n=3. Revisa integra_phi.")

        # exactas del pozo en los mismos puntos
        ex1 = well_wave_exact(1, x, L)
        ex2 = well_wave_exact(2, x, L)
        ex3 = well_wave_exact(3, x, L)

        # Normalizamos numéricas
        phi1n = align_sign(l2_normalize(x, phi1), ex1)
        phi2n = align_sign(l2_normalize(x, phi2), ex2)
        phi3n = align_sign(l2_normalize(x, phi3), ex3)

        # plot: 3 paneles, 2 curvas por panel
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        axes[0].plot(x, phi1n, label="Numérica")
        axes[0].plot(x, ex1, "--", label="Exacta pozo infinito")
        axes[0].set_ylabel(r"$\phi_1(x)$")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(x, phi2n, label="Numérica")
        axes[1].plot(x, ex2, "--", label="Exacta pozo infinito")
        axes[1].set_ylabel(r"$\phi_2(x)$")
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(x, phi3n, label="Numérica")
        axes[2].plot(x, ex3, "--", label="Exacta pozo infinito")
        axes[2].set_ylabel(r"$\phi_3(x)$")
        axes[2].set_xlabel("x")
        axes[2].grid(True)
        axes[2].legend()

        axes[0].set_title(rf"Comparación con pozo infinito (L={L}, h={h}, $\alpha$={alpha:g})")

        fig_path = os.path.join(out_dir, f"phi_vs_pozo_infinito_alpha_{alpha:.0e}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Figura guardada: {fig_path}")


if __name__ == "__main__":
    main()
