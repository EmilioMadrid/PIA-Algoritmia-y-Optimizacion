# Proyecto Final: Hiper-heurística de Selección para TSP

Proyecto integrador de **Algoritmia y Optimización** enfocado en construir una hiper-heurística de selección para el Problema del Viajante (TSP) sobre instancias en $[0, 1]^2$. Dado un conjunto de características de la instancia, un clasificador predice cuál de tres heurísticas/metaheurísticas de la Tarea 2 obtendrá el menor costo de recorrido:

- **Búsqueda Local 2-opt (`2opt_hill_climbing`)**
- **Recocido Simulado + 2-opt (`sa_2opt`)**
- **Árbol de Expansión Mínima + refinamiento 2-opt (`mst_2opt`)**

---

## Estructura del Repositorio

```text
PIA-Algoritmia-y-Optimizacion/
├── hh_tsp_dataset/           # Dataset original (train.csv, test.csv e instances/)
├── src/
│   ├── __init__.py
│   ├── verificar_datos.py    # Resolución dinámica de rutas y verificación de instancias
│   ├── generar_etiquetas.py  # Ejecución de las 3 heurísticas y etiquetado de train/test
│   └── entrenar_ml.py        # Entrenamiento del Árbol de Decisión, métricas y gráficas
├── outputs/                  # CSV etiquetados, métricas y gráficas generadas
├── .gitignore
└── README.md
```

---

## Requisitos

- **Python:** Versión 3.10 o superior (probado en Python 3.13).
- **Librerías de Python:** `numpy`, `pandas`, `matplotlib`, `seaborn` y `scikit-learn`.

Instala todas las dependencias necesarias ejecutando desde la terminal:

```bash
py -m pip install numpy pandas matplotlib seaborn scikit-learn
```

*(En sistemas Linux/macOS o entornos donde `python` esté en el `PATH`, puedes usar `python -m pip install ...`).*

---

## Cómo Reproducir el Proyecto

Clona el repositorio, abre una terminal en la carpeta raíz (`PIA-Algoritmia-y-Optimizacion`) y ejecuta los scripts en el siguiente orden:

### 1. Verificar carga de datos e instancias
Comprueba que las rutas dinámicas detecten correctamente `train.csv` (351 filas), `test.csv` (149 filas) y la carpeta `instances/`:

```bash
py -m src.verificar_datos
```

### 2. Generar etiquetas de entrenamiento y prueba (Parte 1)
Ejecuta los tres algoritmos sobre las 500 instancias para registrar sus costos y determinar el algoritmo ganador (`best_algorithm`):

```bash
py -m src.generar_etiquetas
```

- **Archivos generados:** `outputs/train_labeled.csv` (351 filas) y `outputs/test_labeled.csv` (149 filas) con las columnas adicionales `cost_2opt_hill_climbing`, `cost_sa_2opt`, `cost_mst_2opt` y `best_algorithm`.
- **Parámetros documentados de `sa_2opt`:** `T0 = 100.0`, `T_min = 1e-3`, `alpha = 0.985`, `iter_max = 1500` (con refinamiento local 2-opt al finalizar).

### 3. Entrenar y evaluar el modelo de Machine Learning (Parte 2)
Entrena el Árbol de Decisión sobre `train_labeled.csv`, compara su desempeño contra el *baseline* trivial en `test_labeled.csv` y genera las visualizaciones:

```bash
py -m src.entrenar_ml
```

- **Salidas generadas:** Reporte en consola de balance de clases, *accuracy*, matriz de confusión, *regret* promedio, importancia de características, tablas por tamaño/distribución y la figura `outputs/evaluacion_ml.png`.

---

## Resumen de Resultados

- **Balance de clases (500 instancias):** `sa_2opt` (34.40%), `mst_2opt` (34.20%), `2opt_hill_climbing` (31.40%).
- **Desempeño en `test` (149 instancias):**
  - **Accuracy Baseline trivial (`sa_2opt`):** 32.89% | **Regret promedio:** 0.0666
  - **Accuracy Árbol de Decisión (`max_depth=5`, `min_samples_leaf=8`):** **45.64%** | **Regret promedio:** **0.0507**
- **Características más influyentes:** `n` (0.3474), `std_coords` (0.2711), `mean_nn_dist` (0.1516), `mean_pairwise_dist` (0.1413) y `mean_centroid_dist` (0.0886).

---

## Integrantes

- Emilio Hernández Madrid (2086253)
- Emiliano Reyna Salinas (2145707)