import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.verificar_datos import resolver_rutas_proyecto

FEATURES_NUM = [
    "n",
    "mean_pairwise_dist",
    "std_coords",
    "mean_centroid_dist",
    "mean_nn_dist",
]
FEATURES_CAT = ["distribution"]
TARGET_COL = "best_algorithm"

COST_COLS = {
    "2opt_hill_climbing": "cost_2opt_hill_climbing",
    "sa_2opt": "cost_sa_2opt",
    "mst_2opt": "cost_mst_2opt",
}


def calcular_regret_test(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.Series:
    """
    Para cada instancia de prueba, calcula:
    regret = costo(algoritmo predicho por el modelo) - costo(mejor algoritmo real)
    """
    regrets = []
    for (_, fila), pred_alg in zip(test_df.iterrows(), y_pred):
        col_pred = COST_COLS[pred_alg]
        costo_elegido = float(fila[col_pred])
        costo_optimo_real = min(
            float(fila["cost_2opt_hill_climbing"]),
            float(fila["cost_sa_2opt"]),
            float(fila["cost_mst_2opt"]),
        )
        regrets.append(max(0.0, costo_elegido - costo_optimo_real))
    return pd.Series(regrets, index=test_df.index, name="regret")


def main() -> None:
    _, _, output_dir = resolver_rutas_proyecto()
    train_path = output_dir / "train_labeled.csv"
    test_path = output_dir / "test_labeled.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "No se encontraron train_labeled.csv y test_labeled.csv en outputs/. "
            "Ejecuta primero `py -m src.generar_etiquetas`."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURES_NUM + FEATURES_CAT]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURES_NUM + FEATURES_CAT]
    y_test = test_df[TARGET_COL]

    # 1. Balance de clases en 500 instancias
    completo = pd.concat([train_df, test_df], ignore_index=True)
    print("=== BALANCE DE CLASES (500 INSTANCIAS) ===")
    print(completo[TARGET_COL].value_counts(normalize=True).mul(100).round(2))

    # 2. Baseline trivial: siempre predecir la clase más frecuente en train
    clase_mayoritaria = y_train.mode()[0]
    y_pred_baseline = np.array([clase_mayoritaria] * len(y_test))
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    regret_baseline = calcular_regret_test(test_df, y_pred_baseline)

    # 3. Clasificador: Árbol de Decisión
    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
            ("num", "passthrough", FEATURES_NUM),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("prep", preprocesador),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=2086253)),
        ]
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc_modelo = accuracy_score(y_test, y_pred)
    regret_modelo = calcular_regret_test(test_df, y_pred)

    print("\n=== EVALUACIÓN EN TEST ===")
    print(f"Clase mayoritaria (baseline): {clase_mayoritaria}")
    print(f"Accuracy Baseline:            {acc_baseline:.4f} ({acc_baseline * 100:.2f}%)")
    print(f"Accuracy Árbol de Decisión:   {acc_modelo:.4f} ({acc_modelo * 100:.2f}%)")
    print(f"Regret promedio Baseline:     {regret_baseline.mean():.4f}")
    print(f"Regret promedio Modelo:       {regret_modelo.mean():.4f}")

    print("\n=== REPORTE DE CLASIFICACIÓN ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 4. Importancia de características
    ohe_cols = list(
        modelo.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(FEATURES_CAT)
    )
    nombres_features = ohe_cols + FEATURES_NUM
    importancias = modelo.named_steps["clf"].feature_importances_
    df_imp = (
        pd.DataFrame({"feature": nombres_features, "importance": importancias})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("=== IMPORTANCIA DE CARACTERÍSTICAS ===")
    print(df_imp.round(4))

    # 5. Patrones por distribución y tamaño
    print("\n=== GANADOR POR DISTRIBUCIÓN (500 INSTANCIAS) ===")
    print(pd.crosstab(completo["distribution"], completo[TARGET_COL], normalize="index").mul(100).round(2))

    print("\n=== GANADOR POR TAMAÑO n (500 INSTANCIAS) ===")
    print(pd.crosstab(completo["n"], completo[TARGET_COL], normalize="index").mul(100).round(2))

    # 6. Matriz de confusión e importancia en gráficas
    etiquetas_orden = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]
    cm = confusion_matrix(y_test, y_pred, labels=etiquetas_orden)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas_orden)
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Matriz de Confusión (Test)")

    top_imp = df_imp.head(8).iloc[::-1]
    axes[1].barh(top_imp["feature"], top_imp["importance"], color="#2b6cb0")
    axes[1].set_title("Importancia de Características (Árbol)")
    axes[1].set_xlabel("Importancia")

    plt.tight_layout()
    ruta_fig = output_dir / "evaluacion_ml.png"
    plt.savefig(ruta_fig, dpi=300)
    plt.close(fig)
    print(f"\nGráfica guardada en: {ruta_fig}")


if __name__ == "__main__":
    main()