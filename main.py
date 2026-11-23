import os
import numpy as np
import matplotlib.pyplot as plt
from potencial import energia_estado_n, integra_phi
from writer import (
    init_summary_csv, append_summary_row,
    init_wave_samples_csv_for_L, append_wave_samples_row,
    sample_phi_at_points
)

def x_points_for_L(L: float):
    offsets = np.arange(0.1, L, 0.2)
    left = -L + offsets
    right = L - offsets[::-1]
    return list(left) + list(right)

def main():
    alpha = 2.0
    L_values = [1.0, 2.0, 3.0]

    # Mallados para el estudio de convergencia
    h_values = [0.1, 0.05, 0.02, 0.01, 0.005]

    # Referencia
    h_ref = 0.001

    out_dir = "resultados_pozo_parabolico"
    os.makedirs(out_dir, exist_ok=True)

    summary_path = init_summary_csv(out_dir, "resumen_energias_alpha2.csv")

    for L in L_values:
        print(f"\n=== L={L}  alpha={alpha} ===")

        x_points = x_points_for_L(L) #puntos x


        # CSV único de muestras para este L
        wave_samples_path = init_wave_samples_csv_for_L(out_dir, L, alpha, x_points)
        wave_samples_name = os.path.basename(wave_samples_path)

        # E inicial
        omega = np.sqrt(2.0 * alpha)
        guess = [0.5 * omega, 1.5 * omega, 2.5 * omega]

        # -------------------------
        # 1) Referencia
        # -------------------------
        deltas_ref = [1.0, 2.0, 3.0]
        Eref = []
        for n in (1, 2, 3):
            En = energia_estado_n(
                n, L, alpha, h_ref,
                eps_inicial=guess[n - 1],
                delta_eps_inicial=deltas_ref[n - 1],
            )
            Eref.append(En)

        E1_ref, E2_ref, E3_ref = Eref
        print(f"[REF] h={h_ref}  E1={E1_ref:.10f}  E2={E2_ref:.10f}  E3={E3_ref:.10f}")

        # energías ref en resumen
        append_summary_row(summary_path, L, alpha, h_ref, E1_ref, E2_ref, E3_ref, 1, wave_samples_name)

        # muestras de phi_n(x_k) en el CSV del L (3 filas: n=1,2,3)
        for n, En in enumerate(Eref, start=1):
            xs, phi = integra_phi(En, L, alpha, h_ref)
            #phi = normalize_phi(xs, phi)
            samples = sample_phi_at_points(xs, phi, x_points)
            append_wave_samples_row(wave_samples_path, L, alpha, h_ref, 1, n, En, samples)

        # -------------------------
        # 2) Barrido en h_values
        # -------------------------
        hs = []
        E1s, E2s, E3s = [], [], []

        guess = Eref[:]         # mejor seed
        deltas = [1.0, 1.0, 1.0]

        for h in h_values:
            Es = []
            for n in (1, 2, 3):
                En = energia_estado_n(
                    n, L, alpha, h,
                    eps_inicial=guess[n - 1],
                    delta_eps_inicial=deltas[n - 1],
                )
                Es.append(En)

            E1, E2, E3 = Es
            guess = Es[:]

            hs.append(h)
            E1s.append(E1)
            E2s.append(E2)
            E3s.append(E3)

            print(f"h={h:7.4f}  E1={E1:14.10f}  E2={E2:14.10f}  E3={E3:14.10f}")

            # energías en resumen
            append_summary_row(summary_path, L, alpha, h, E1, E2, E3, 0, wave_samples_name)

            # muestras de phi_n en el CSV del L
            for n, En in enumerate(Es, start=1):
                xs, phi = integra_phi(En, L, alpha, h)
                samples = sample_phi_at_points(xs, phi, x_points)
                append_wave_samples_row(wave_samples_path, L, alpha, h, 0, n, En, samples)

        # -------------------------
        # 3) Plot convergencia (log10(h) vs log10(error))
        # -------------------------
        fig_path = plot_convergence_loglog(
            out_dir, L, alpha, h_ref,
            hs, E1s, E2s, E3s,
            E1_ref, E2_ref, E3_ref
        )
        print(f"Figura guardada: {fig_path}")
        print(f"CSV wave-samples (solo para L={L}): {wave_samples_path}")

    print(f"\nResumen guardado en: {summary_path}")
    print(f"Carpeta de resultados: {out_dir}")

def plot_convergence_loglog(out_dir: str, L: float, alpha: float, h_ref: float,
                            hs: list[float],
                            E1s: list[float], E2s: list[float], E3s: list[float],
                            E1_ref: float, E2_ref: float, E3_ref: float) -> str:
    #estudios de convergencia
    os.makedirs(out_dir, exist_ok=True)

    # errores vs referencia
    err1 = [abs(e - E1_ref) for e in E1s]
    err2 = [abs(e - E2_ref) for e in E2s]
    err3 = [abs(e - E3_ref) for e in E3s]

    floor = 1e-16
    logh = [np.log10(h) for h in hs]
    logerr1 = [np.log10(max(e, floor)) for e in err1]
    logerr2 = [np.log10(max(e, floor)) for e in err2]
    logerr3 = [np.log10(max(e, floor)) for e in err3]

    plt.figure()
    plt.plot(logh, logerr1, marker="o", label="E1")
    plt.plot(logh, logerr2, marker="o", label="E2")
    plt.plot(logh, logerr3, marker="o", label="E3")
    plt.xlabel("log10(h)")
    plt.ylabel("log10(|E(h) - E(h_ref)|)")
    plt.title(f"Convergencia (L={L}, alpha={alpha})  referencia h_ref={h_ref}")
    plt.grid(True)
    plt.legend()

    fig_path = os.path.join(out_dir, f"convergencia_L{L:g}_alpha{alpha:g}.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    return fig_path

if __name__ == "__main__":
    main()
