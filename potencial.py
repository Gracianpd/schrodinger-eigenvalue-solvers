import numpy as np


def V_pozo(x, alpha):
    #funcion que calcula el potencial en un punto x
    return alpha * x * x


def nodos_por_y(E, L, alpha, h, y2_grande=1e30, eps_y=1e-14):
    #funcion que cuenta numero de nodos en cada nivel energetico
    N = int(round((2 * L) / h))
    if N < 3:
        raise ValueError("h demasiado grande: necesitas más puntos.")
    h_eff = (2 * L) / N
    xs = np.linspace(-L, L, N + 1)

    y_prev = float(y2_grande)

    #asumimos signo positivo al empezar
    s_phi = 1
    nodos = 0

    for i in range(1, N):
        Vi = alpha * xs[i] * xs[i]

        y_safe = y_prev
        if abs(y_safe) < eps_y:
            y_safe = np.sign(y_safe) * eps_y if y_safe != 0 else eps_y

        y_next = (2.0 + (Vi - E) * h_eff * h_eff) - 1.0 / y_safe

        # reconstrucción del signo de phi y conteo de cambios de signo en phi
        s_y = 1 if y_next >= 0 else -1
        s_phi_next = s_y * s_phi
        if s_phi_next != s_phi:
            nodos += 1
        s_phi = s_phi_next

        y_prev = y_next
        if not np.isfinite(y_prev):
            break

    return nodos


def energia_estado_n(n, L, alpha, h,
                     eps_inicial,
                     delta_eps_inicial=1.0,
                     max_iter=2000):
    #funcion que calcula la energia de un estado n mediante el metodo de la biseccion
    target = n - 1
    eps = float(eps_inicial)
    delta = float(delta_eps_inicial)

    # --- Paso 1: construir bracket con nodos ---
    nod = nodos_por_y(eps, L, alpha, h)

    while nod > target:
        eps -= delta
        if eps < 0:
            eps = 0.0
            break
        nod = nodos_por_y(eps, L, alpha, h)

    E_low = eps
    nod_low = nodos_por_y(E_low, L, alpha, h)
    if nod_low > target:
        raise RuntimeError("No se pudo encontrar E_low con nodos <= target.")

    E_high = E_low
    while True:
        E_high += delta
        nod_high = nodos_por_y(E_high, L, alpha, h)
        if nod_high > target:
            break
        if E_high > 1e6:
            raise RuntimeError("No se encontró el salto de nodos: aumenta delta_eps_inicial o revisa parámetros.")

    # --- Paso 2: bisección por nodos - criterio de parada entre dos numeros de nodos ---
    for _ in range(max_iter):
        nod_low = nodos_por_y(E_low, L, alpha, h)
        nod_high = nodos_por_y(E_high, L, alpha, h)

        # aseguramos que realmente encierra el salto
        if not (nod_low <= target and nod_high > target):
            raise RuntimeError("El bracket dejó de encerrar el salto de nodos (revisar nodos_por_y).")

        E_mid = 0.5 * (E_low + E_high)
        nod_mid = nodos_por_y(E_mid, L, alpha, h)

        # Caso 1: el punto medio sigue en la región baja ->  subir E_low
        if nod_mid <= target:
            E_low = E_mid
            continue

        # Caso 2: el punto medio ya está en la región alta ->  bajar E_high
        if nod_mid > target:
            E_high = E_mid
            continue

    return 0.5 * (E_low + E_high)


def integra_phi(E, L, alpha, h, phi0=0.0, phi1=1e-5):
    #obtiene mediante metodo iterativo la autofuncion
    N = int(round((2 * L) / h))
    if N < 3:
        raise ValueError("h demasiado grande.")
    h_eff = (2 * L) / N
    xs = np.linspace(-L, L, N + 1)

    phi = np.zeros_like(xs)
    phi[0] = phi0
    phi[1] = phi1

    for i in range(1, len(xs) - 1):
        Vi = V_pozo(xs[i], alpha)
        phi[i + 1] = (((Vi - E) * h_eff * h_eff) + 2.0) * phi[i] - phi[i - 1]
        if not np.isfinite(phi[i + 1]):
            break

    return xs, phi
