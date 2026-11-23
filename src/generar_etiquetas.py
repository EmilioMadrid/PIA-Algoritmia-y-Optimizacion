import math
import random
import numpy as np
import pandas as pd

from src.verificar_datos import cargar_coordenadas, resolver_rutas_proyecto

# ==============================================================================
# PARÁMETROS DOCUMENTADOS DE RECOCIDO SIMULADO (SA + 2-OPT)
# ==============================================================================
SA_PARAMS = {
    "T0": 100.0,
    "T_min": 1e-3,
    "alpha": 0.985,
    "iter_max": 1500,
}


def matriz_distancias(coords: np.ndarray) -> np.ndarray:
    """Calcula la matriz de distancias euclidianas N x N."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)


def costo_tour_matriz(tour: list[int], dist: np.ndarray) -> float:
    """Calcula la longitud total de un tour cerrado usando la matriz de distancias."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += float(dist[tour[i], tour[(i + 1) % n]])
    return total


def aplicar_2opt_swap(tour: list[int], i: int, j: int) -> list[int]:
    """Invierte el segmento entre los índices i y j (inclusive)."""
    return tour[:i] + list(reversed(tour[i : j + 1])) + tour[j + 1 :]


def delta_2opt(tour: list[int], i: int, j: int, dist: np.ndarray) -> float:
    """Calcula el cambio neto de costo al hacer un movimiento 2-opt entre i y j."""
    n = len(tour)
    a = tour[(i - 1) % n]
    b = tour[i]
    c = tour[j]
    d = tour[(j + 1) % n]

    antes = float(dist[a, b] + dist[c, d])
    despues = float(dist[a, c] + dist[b, d])
    return despues - antes


# ==============================================================================
# 1. BÚSQUEDA LOCAL: 2-OPT HILL CLIMBING
# ==============================================================================
def run_2opt_hill_climbing(
    coords: np.ndarray, dist: np.ndarray | None = None, seed: int | None = None
) -> float:
    """Búsqueda local 2-opt (first/best improvement hasta óptimo local)."""
    n = len(coords)
    if n <= 2:
        return 0.0

    rng = random.Random(seed)
    tour = list(range(n))
    rng.shuffle(tour)

    if dist is None:
        dist = matriz_distancias(coords)

    mejora = True
    while mejora:
        mejora = False
        mejor_delta = -1e-12
        mejor_par: tuple[int, int] | None = None

        for i in range(n - 1):
            j_inicio = i + 2 if i == 0 else i + 1
            j_fin = n - 1 if i == 0 else n
            for j in range(j_inicio, j_fin):
                d = delta_2opt(tour, i, j, dist)
                if d < mejor_delta:
                    mejor_delta = d
                    mejor_par = (i, j)

        if mejor_par is not None:
            i, j = mejor_par
            tour = aplicar_2opt_swap(tour, i, j)
            mejora = True

    return costo_tour_matriz(tour, dist)


# ==============================================================================
# 2. RECOCIDO SIMULADO (SA) + 2-OPT
# ==============================================================================
def run_sa_2opt(
    coords: np.ndarray,
    dist: np.ndarray | None = None,
    seed: int | None = None,
    T0: float = SA_PARAMS["T0"],
    T_min: float = SA_PARAMS["T_min"],
    alpha: float = SA_PARAMS["alpha"],
    iter_max: int = SA_PARAMS["iter_max"],
) -> float:
    """Recocido Simulado con vecindario 2-opt."""
    n = len(coords)
    if n <= 2:
        return 0.0

    rng = random.Random(seed)
    actual = list(range(n))
    rng.shuffle(actual)

    if dist is None:
        dist = matriz_distancias(coords)

    costo_actual = costo_tour_matriz(actual, dist)
    mejor_costo = costo_actual

    temp = T0
    for _ in range(iter_max):
        if temp <= T_min:
            break

        i = rng.randrange(0, n - 1)
        j_inicio = i + 2 if i == 0 else i + 1
        j_fin = n - 2 if i == 0 else n - 1
        if j_inicio > j_fin:
            continue
        j = rng.randint(j_inicio, j_fin)

        d = delta_2opt(actual, i, j, dist)
        aceptar = False
        if d < 0:
            aceptar = True
        else:
            prob = math.exp(-d / max(temp, 1e-12))
            if rng.random() < prob:
                aceptar = True

        if aceptar:
            actual = aplicar_2opt_swap(actual, i, j)
            costo_actual += d
            if costo_actual < mejor_costo:
                mejor_costo = costo_actual

        temp *= alpha

    # Refinamiento local final con 2-opt hill climbing
    # (se aplica sobre una copia de la mejor ruta encontrada)
    mejor_tour = list(range(n))
    rng.shuffle(mejor_tour)
    # Partimos desde 'actual' refinado con 2-opt si mejora
    mejora = True
    while mejora:
        mejora = False
        for i in range(n - 1):
            j_inicio = i + 2 if i == 0 else i + 1
            j_fin = n - 1 if i == 0 else n
            for j in range(j_inicio, j_fin):
                d = delta_2opt(actual, i, j, dist)
                if d < -1e-12:
                    actual = aplicar_2opt_swap(actual, i, j)
                    costo_actual += d
                    if costo_actual < mejor_costo:
                        mejor_costo = costo_actual
                    mejora = True
                    break
            if mejora:
                break

    return float(mejor_costo)


# ==============================================================================
# 3. APROXIMACIÓN MST (PREORDEN DFS) + 2-OPT
# ==============================================================================
def construir_tour_mst_dfs(dist: np.ndarray, inicio: int = 0) -> list[int]:
    """Construye un tour inicial mediante recorrido preorden sobre un MST de Prim."""
    n = dist.shape[0]
    visitado = [False] * n
    clave = [math.inf] * n
    padre = [-1] * n

    clave[inicio] = 0.0
    adyacencia: list[list[int]] = [[] for _ in range(n)]

    for _ in range(n):
        u = -1
        mejor_clave = math.inf
        for v in range(n):
            if not visitado[v] and clave[v] < mejor_clave:
                mejor_clave = clave[v]
                u = v

        if u == -1:
            break
        visitado[u] = True
        if padre[u] != -1:
            adyacencia[padre[u]].append(u)
            adyacencia[u].append(padre[u])

        for v in range(n):
            if not visitado[v] and dist[u, v] < clave[v]:
                clave[v] = float(dist[u, v])
                padre[v] = u

    # Recorrido preorden (DFS)
    tour: list[int] = []
    pila = [inicio]
    vistos_dfs = set()
    while pila:
        nodo = pila.pop()
        if nodo in vistos_dfs:
            continue
        vistos_dfs.add(nodo)
        tour.append(nodo)
        for vecino in reversed(adyacencia[nodo]):
            if vecino not in vistos_dfs:
                pila.append(vecino)

    return tour


def run_mst_2opt(
    coords: np.ndarray, dist: np.ndarray | None = None, seed: int | None = None
) -> float:
    """Tour inicial por MST + refinamiento 2-opt."""
    n = len(coords)
    if n <= 2:
        return 0.0

    if dist is None:
        dist = matriz_distancias(coords)

    inicio = 0 if seed is None else (abs(seed) % n)
    tour = construir_tour_mst_dfs(dist, inicio=inicio)

    mejora = True
    while mejora:
        mejora = False
        mejor_delta = -1e-12
        mejor_par: tuple[int, int] | None = None

        for i in range(n - 1):
            j_inicio = i + 2 if i == 0 else i + 1
            j_fin = n - 1 if i == 0 else n
            for j in range(j_inicio, j_fin):
                d = delta_2opt(tour, i, j, dist)
                if d < mejor_delta:
                    mejor_delta = d
                    mejor_par = (i, j)

        if mejor_par is not None:
            i, j = mejor_par
            tour = aplicar_2opt_swap(tour, i, j)
            mejora = True

    return costo_tour_matriz(tour, dist)


# ==============================================================================
# ETIQUETADO DEL DATASET (PARTE 1)
# ==============================================================================
def etiquetar_dataset(df: pd.DataFrame, instances_dir, seed_base: int = 2086253) -> pd.DataFrame:
    df_etiquetado = df.copy()
    costos_hc: list[float] = []
    costos_sa: list[float] = []
    costos_mst: list[float] = []
    mejores_algoritmos: list[str] = []

    for idx, fila in df_etiquetado.iterrows():
        instance_id = fila["instance_id"]
        coords_df = cargar_coordenadas(instance_id, instances_dir)

        if {"x", "y"}.issubset(coords_df.columns):
            coords = coords_df[["x", "y"]].to_numpy(dtype=float)
        else:
            coords = coords_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)

        dist = matriz_distancias(coords)
        seed_inst = seed_base + int(idx)

        c_hc = run_2opt_hill_climbing(coords, dist=dist, seed=seed_inst)
        c_sa = run_sa_2opt(coords, dist=dist, seed=seed_inst + 10_000)
        c_mst = run_mst_2opt(coords, dist=dist, seed=seed_inst + 20_000)

        costos_por_alg = {
            "2opt_hill_climbing": c_hc,
            "sa_2opt": c_sa,
            "mst_2opt": c_mst,
        }
        mejor = min(costos_por_alg, key=costos_por_alg.get)

        costos_hc.append(c_hc)
        costos_sa.append(c_sa)
        costos_mst.append(c_mst)
        mejores_algoritmos.append(mejor)

    df_etiquetado["cost_2opt_hill_climbing"] = costos_hc
    df_etiquetado["cost_sa_2opt"] = costos_sa
    df_etiquetado["cost_mst_2opt"] = costos_mst
    df_etiquetado["best_algorithm"] = mejores_algoritmos
    return df_etiquetado


def main() -> None:
    data_dir, instances_dir, output_dir = resolver_rutas_proyecto()
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print("Etiquetando train.csv y test.csv sobre las instancias TSP...")
    train_labeled = etiquetar_dataset(train_df, instances_dir, seed_base=2086253)
    test_labeled = etiquetar_dataset(test_df, instances_dir, seed_base=102086253)

    ruta_train_out = output_dir / "train_labeled.csv"
    ruta_test_out = output_dir / "test_labeled.csv"

    train_labeled.to_csv(ruta_train_out, index=False)
    test_labeled.to_csv(ruta_test_out, index=False)

    completo = pd.concat([train_labeled, test_labeled], ignore_index=True)
    print(f"\nGuardado: {ruta_train_out} ({len(train_labeled)} filas)")
    print(f"Guardado: {ruta_test_out} ({len(test_labeled)} filas)")
    print("\nBalance de clases en las 500 instancias (train + test):")
    print(completo["best_algorithm"].value_counts(normalize=True).mul(100).round(2))


if __name__ == "__main__":
    main()