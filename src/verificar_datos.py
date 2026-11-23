from pathlib import Path
import pandas as pd


def obtener_raiz_proyecto() -> Path:
    """Detecta la carpeta raíz del proyecto (donde viven hh_tsp_dataset, src y outputs)."""
    base_script = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidatos = [
        base_script,
        base_script.parent,
        base_script.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]

    for ruta in candidatos:
        resuelta = ruta.resolve()
        if (resuelta / "hh_tsp_dataset").is_dir() and (resuelta / "src").is_dir():
            return resuelta

    return base_script.parent if base_script.name == "src" else base_script


def resolver_rutas_proyecto() -> tuple[Path, Path, Path]:
    """Devuelve (data_dir, instances_dir, output_dir) usando rutas dinámicas."""
    raiz = obtener_raiz_proyecto()
    candidatos_data = [
        raiz / "hh_tsp_dataset" / "hh_tsp_dataset" / "data",
        raiz / "hh_tsp_dataset" / "data",
        raiz / "data",
    ]

    data_dir = None
    for cand in candidatos_data:
        if (cand / "train.csv").exists() and (cand / "test.csv").exists() and (cand / "instances").is_dir():
            data_dir = cand.resolve()
            break

    if data_dir is None:
        raise FileNotFoundError(
            f"No se encontró la carpeta `data` con train.csv, test.csv e instances/ desde: {raiz}"
        )

    instances_dir = data_dir / "instances"
    output_dir = (raiz / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, instances_dir, output_dir


def cargar_coordenadas(instance_id: str, instances_dir: Path) -> pd.DataFrame:
    """Carga las coordenadas (x, y) de una instancia dada su instance_id."""
    nombre = str(instance_id).strip()
    if not nombre.endswith(".csv"):
        nombre = f"{nombre}.csv"

    ruta_archivo = instances_dir / nombre
    if not ruta_archivo.exists():
        raise FileNotFoundError(f"No se encontró el archivo de instancia: {ruta_archivo}")

    return pd.read_csv(ruta_archivo)


if __name__ == "__main__":
    data_dir, instances_dir, output_dir = resolver_rutas_proyecto()
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print(f"DATA_DIR:      {data_dir}")
    print(f"INSTANCES_DIR: {instances_dir}")
    print(f"OUTPUT_DIR:    {output_dir}")
    print(f"Train filas:   {len(train_df)} | Columnas: {list(train_df.columns)}")
    print(f"Test filas:    {len(test_df)} | Columnas: {list(test_df.columns)}")

    primera_id = train_df.iloc[0]["instance_id"]
    coords_ejemplo = cargar_coordenadas(primera_id, instances_dir)
    print(f"\nVerificación instancia '{primera_id}' ({len(coords_ejemplo)} filas):")
    print(coords_ejemplo.head())