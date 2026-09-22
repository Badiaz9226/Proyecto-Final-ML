"""Prueba de integración del artefacto entrenado y la aplicación Streamlit.

La comprobación es deliberadamente pequeña: valida el contrato de 18 variables,
carga el pipeline serializado y realiza una predicción con una presentación real
del conjunto modelado. El resultado queda registrado como CSV para que la
verificación sea auditable tanto en local como en Google Colab.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Permite ejecutar tanto ``python -m src.integration_check`` como
# ``python src/integration_check.py`` desde la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Necesario para que joblib pueda reconstruir el tokenizador serializado.
from src.transformers import split_pipe_tokens as _split_pipe_tokens  # noqa: F401


NUMERIC_FEATURES = [
    "cantidad_cum",
    "numero_filas_fuente",
    "numero_cantidades_ingrediente_distintas",
    "numero_principios_activos",
    "numero_vias_administracion",
    "numero_codigos_atc",
    "numero_unidades_medida",
    "numero_unidades_referencia",
    "numero_tipos_rol",
    "numero_actores",
]

CATEGORICAL_FEATURES = [
    "forma_farmaceutica",
    "codigo_concentracion",
    "unidad_cum",
    "modalidad",
    "vias_administracion",
    "unidades_medida",
    "familias_atc",
    "tipos_rol",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_FILES = [
    Path("app.py"),
    Path("requirements.txt"),
    Path("runtime.txt"),
    Path("README.md"),
    Path("notebooks/Proyecto_Final_ML_CUM.ipynb"),
    Path("reports/reporte_ydata_profiling.html"),
    Path("models/pipeline_muestra_medica.joblib"),
    Path("models/metadata_modelo.json"),
    Path("data/dataset_modelado_cum.csv"),
    Path("reports/resultados_modelado.json"),
]

EXPECTED_APP_PATHS = {
    "MODEL_PATH": Path("models/pipeline_muestra_medica.joblib"),
    "METADATA_PATH": Path("models/metadata_modelo.json"),
    "RESULTS_PATH": Path("reports/resultados_modelado.json"),
    "CONFUSION_FIGURE": Path("figures/matriz_confusion_test.png"),
    "COMPARISON_FIGURE": Path("figures/comparacion_modelos_cv.png"),
}


def _record(
    rows: list[dict[str, Any]],
    check: str,
    passed: bool,
    detail: str,
) -> None:
    rows.append(
        {
            "comprobacion": check,
            "estado": "OK" if passed else "ERROR",
            "detalle": detail,
        }
    )


def _literal_path(node: ast.AST) -> str | None:
    """Extract ``APP_DIR / 'folder' / 'file'`` as a relative POSIX path."""

    parts: list[str] = []
    current = node
    while isinstance(current, ast.BinOp) and isinstance(current.op, ast.Div):
        if not isinstance(current.right, ast.Constant) or not isinstance(
            current.right.value, str
        ):
            return None
        parts.append(current.right.value)
        current = current.left
    if not isinstance(current, ast.Name) or current.id != "APP_DIR":
        return None
    return Path(*reversed(parts)).as_posix()


def _read_app_paths(app_path: Path) -> dict[str, str]:
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    observed: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            value = node.value
        else:
            if not isinstance(node.target, ast.Name):
                continue
            name = node.target.id
            value = node.value
        if value is not None and name in EXPECTED_APP_PATHS:
            path = _literal_path(value)
            if path is not None:
                observed[name] = path
    return observed


def run_checks() -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []

    missing = [str(path) for path in REQUIRED_FILES if not (PROJECT_ROOT / path).is_file()]
    _record(
        rows,
        "Archivos obligatorios",
        not missing,
        "Todos están presentes" if not missing else f"Faltan: {', '.join(missing)}",
    )

    # Las validaciones siguientes dependen de los artefactos de entrenamiento.
    if missing:
        return pd.DataFrame(rows), False

    metadata_path = PROJECT_ROOT / "models" / "metadata_modelo.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_features = metadata.get("numeric_features", []) + metadata.get(
        "categorical_features", []
    )
    feature_contract_ok = metadata_features == MODEL_FEATURES and len(MODEL_FEATURES) == 18
    _record(
        rows,
        "Contrato exacto de 18 variables",
        feature_contract_ok,
        (
            "Orden y nombres coinciden con la aplicación"
            if feature_contract_ok
            else f"Esperadas: {MODEL_FEATURES}; recibidas: {metadata_features}"
        ),
    )

    app_path = PROJECT_ROOT / "app.py"
    app_tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    app_literals: dict[str, Any] = {}
    for node in app_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "NUMERIC_FEATURES",
                "SINGLE_CATEGORICAL_FEATURES",
                "MULTI_CATEGORICAL_FEATURES",
            }:
                app_literals[target.id] = ast.literal_eval(node.value)
    app_features = (
        app_literals.get("NUMERIC_FEATURES", [])
        + app_literals.get("SINGLE_CATEGORICAL_FEATURES", [])
        + app_literals.get("MULTI_CATEGORICAL_FEATURES", [])
    )
    app_contract_ok = app_features == MODEL_FEATURES
    _record(
        rows,
        "Contrato aplicación-modelo",
        app_contract_ok,
        "Las 18 entradas conservan el mismo orden" if app_contract_ok else str(app_features),
    )

    app_paths = _read_app_paths(app_path)
    expected_paths = {name: path.as_posix() for name, path in EXPECTED_APP_PATHS.items()}
    paths_ok = app_paths == expected_paths
    _record(
        rows,
        "Rutas portables de la aplicación",
        paths_ok,
        (
            "Todos los artefactos se resuelven desde APP_DIR"
            if paths_ok
            else f"Esperadas: {expected_paths}; recibidas: {app_paths}"
        ),
    )

    data_path = PROJECT_ROOT / "data" / "dataset_modelado_cum.csv"
    sample = pd.read_csv(data_path, nrows=1)
    missing_columns = [column for column in MODEL_FEATURES if column not in sample.columns]
    _record(
        rows,
        "Variables disponibles en el conjunto modelado",
        not missing_columns and len(sample) == 1,
        (
            "Se encontró una presentación válida"
            if not missing_columns and len(sample) == 1
            else f"Columnas faltantes: {missing_columns}; filas leídas: {len(sample)}"
        ),
    )

    model = joblib.load(PROJECT_ROOT / "models" / "pipeline_muestra_medica.joblib")
    input_row = sample.loc[:, MODEL_FEATURES]
    prediction = np.asarray(model.predict(input_row)).reshape(-1)
    prediction_ok = prediction.shape == (1,) and int(prediction[0]) in {0, 1}
    _record(
        rows,
        "Predicción extremo a extremo",
        prediction_ok,
        (
            f"Clase predicha: {'Sí' if int(prediction[0]) == 1 else 'No'}"
            if prediction.size
            else "El pipeline no devolvió una predicción"
        ),
    )

    report = pd.DataFrame(rows)
    return report, bool((report["estado"] == "OK").all())


def main() -> int:
    output_path = PROJECT_ROOT / "reports" / "prueba_integracion.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report, passed = run_checks()
    except Exception as error:  # pragma: no cover - conserva evidencia del fallo
        report = pd.DataFrame(
            [
                {
                    "comprobacion": "Ejecución de la prueba",
                    "estado": "ERROR",
                    "detalle": f"{type(error).__name__}: {error}",
                }
            ]
        )
        passed = False

    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print(f"\nResultado guardado en: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
