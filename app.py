"""Aplicación Streamlit del proyecto final de Aprendizaje de Máquinas.

La aplicación consume el pipeline completo serializado: las transformaciones,
el balanceo aprendido durante el entrenamiento y el clasificador viajan juntos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# El import permite que joblib resuelva la función usada por CountVectorizer.
from src.transformers import split_pipe_tokens as _split_pipe_tokens  # noqa: F401


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "pipeline_muestra_medica.joblib"
METADATA_PATH = APP_DIR / "models" / "metadata_modelo.json"
RESULTS_PATH = APP_DIR / "reports" / "resultados_modelado.json"
CONFUSION_FIGURE = APP_DIR / "figures" / "matriz_confusion_test.png"
COMPARISON_FIGURE = APP_DIR / "figures" / "comparacion_modelos_cv.png"
SOURCE_URL = (
    "https://www.datos.gov.co/Salud-y-Protecci-n-Social/"
    "C-DIGO-NICO-DE-MEDICAMENTOS-VIGENTES/i7cb-raxc/about_data"
)

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

SINGLE_CATEGORICAL_FEATURES = [
    "forma_farmaceutica",
    "codigo_concentracion",
    "unidad_cum",
    "modalidad",
]

MULTI_CATEGORICAL_FEATURES = [
    "vias_administracion",
    "unidades_medida",
    "familias_atc",
    "tipos_rol",
]

CATEGORICAL_FEATURES = SINGLE_CATEGORICAL_FEATURES + MULTI_CATEGORICAL_FEATURES
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABELS = {
    "cantidad_cum": "Cantidad de la presentación",
    "numero_filas_fuente": "Registros asociados en la fuente",
    "numero_cantidades_ingrediente_distintas": "Cantidades de ingrediente distintas",
    "numero_principios_activos": "Principios activos distintos",
    "numero_vias_administracion": "Número de vías de administración",
    "numero_codigos_atc": "Códigos ATC distintos",
    "numero_unidades_medida": "Unidades de medida distintas",
    "numero_unidades_referencia": "Unidades de referencia distintas",
    "numero_tipos_rol": "Tipos de rol distintos",
    "numero_actores": "Actores o establecimientos distintos",
    "forma_farmaceutica": "Forma farmacéutica",
    "codigo_concentracion": "Código de concentración",
    "unidad_cum": "Unidad de la presentación",
    "modalidad": "Modalidad",
    "vias_administracion": "Vías de administración",
    "unidades_medida": "Unidades de medida",
    "familias_atc": "Familias ATC",
    "tipos_rol": "Tipos de rol",
}

HELP_TEXT = {
    "cantidad_cum": "Cantidad declarada para la presentación CUM.",
    "numero_filas_fuente": (
        "Número de filas que originaría esta presentación al combinar ingredientes, "
        "vías, roles y establecimientos."
    ),
    "numero_cantidades_ingrediente_distintas": (
        "Cantidad de valores diferentes informados para los ingredientes."
    ),
    "numero_principios_activos": "Conteo de principios activos diferentes.",
    "numero_vias_administracion": "Debe coincidir con las vías seleccionadas abajo.",
    "numero_codigos_atc": "Conteo de códigos ATC diferentes.",
    "numero_unidades_medida": "Debe coincidir con las unidades seleccionadas abajo.",
    "numero_unidades_referencia": "Conteo de unidades de referencia diferentes.",
    "numero_tipos_rol": "Debe coincidir con los tipos de rol seleccionados abajo.",
    "numero_actores": "Conteo de actores o establecimientos distintos.",
    "codigo_concentracion": "Categoría codificada tal como aparece en la fuente.",
    "familias_atc": "Primera letra de los códigos ATC asociados.",
}

INTEGER_FEATURES = set(NUMERIC_FEATURES) - {"cantidad_cum"}


st.set_page_config(
    page_title="Clasificador CUM · Muestra médica",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cum-navy: #12304a;
            --cum-blue: #176b87;
            --cum-teal: #1f8a70;
            --cum-soft: #eef7f6;
            --cum-amber: #d98418;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 { color: var(--cum-navy); letter-spacing: -0.02em; }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #ffffff 0%, #f5fafb 100%);
            border: 1px solid #dce9ec;
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stForm"] {
            border: 1px solid #dce9ec;
            border-radius: 16px;
            padding: 1.1rem 1.1rem 0.5rem;
            background: #fbfdfd;
        }
        .cum-hero {
            padding: 1.3rem 1.45rem;
            border-radius: 18px;
            color: white;
            background: linear-gradient(120deg, #12304a 0%, #176b87 58%, #1f8a70 100%);
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 24px rgba(18, 48, 74, 0.12);
        }
        .cum-hero h1 { color: white; margin: 0 0 0.35rem; font-size: 2rem; }
        .cum-hero p { margin: 0; opacity: 0.93; max-width: 820px; }
        .cum-result-positive, .cum-result-negative {
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            margin: 0.8rem 0;
        }
        .cum-result-positive {
            background: #fff5e8;
            border-left: 6px solid var(--cum-amber);
        }
        .cum-result-negative {
            background: var(--cum-soft);
            border-left: 6px solid var(--cum-teal);
        }
        .cum-result-positive h3, .cum-result-negative h3 { margin: 0 0 0.25rem; }
        .cum-step {
            min-height: 132px;
            padding: 0.95rem 1rem;
            border: 1px solid #dce9ec;
            border-radius: 14px;
            background: #ffffff;
            margin-bottom: 0.7rem;
        }
        .cum-step strong { color: var(--cum-blue); }
        div[data-baseweb="tab-list"] { overflow-x: auto; }
        @media (max-width: 640px) {
            .block-container { padding: 1rem 0.8rem 2rem; }
            .cum-hero { padding: 1rem; border-radius: 14px; }
            .cum-hero h1 { font-size: 1.55rem; }
            .cum-step { min-height: auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Cargando el pipeline entrenado…")
def load_model(path: str, modified_at: float) -> Any:
    """Load the full preprocessing-and-model pipeline once per artifact version."""

    del modified_at  # It is part of the cache key.
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_json(path: str, modified_at: float) -> dict[str, Any]:
    """Read a JSON artifact and invalidate the cache when it changes."""

    del modified_at
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_if_available(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(str(path), path.stat().st_mtime)


def _validate_metadata(metadata: dict[str, Any]) -> None:
    observed = metadata.get("numeric_features", []) + metadata.get(
        "categorical_features", []
    )
    if observed != MODEL_FEATURES:
        missing = sorted(set(MODEL_FEATURES) - set(observed))
        unexpected = sorted(set(observed) - set(MODEL_FEATURES))
        raise ValueError(
            "Los metadatos no corresponden a las 18 variables esperadas. "
            f"Faltan: {missing or 'ninguna'}. Sobran: {unexpected or 'ninguna'}."
        )


def _numeric_settings(
    feature: str, metadata: dict[str, Any]
) -> tuple[int | float, int | float, int | float, int | float]:
    ranges = metadata.get("numeric_ranges", {}).get(feature, {})
    median = float(ranges.get("median", 1.0) or 0.0)
    observed_max = float(ranges.get("max", max(1.0, median)) or max(1.0, median))
    p99 = float(ranges.get("p99", observed_max) or observed_max)
    sensible_max = max(1.0, observed_max, p99, median)
    if feature in INTEGER_FEATURES:
        return 0, int(np.ceil(sensible_max)), max(0, int(round(median))), 1
    return 0.0, float(sensible_max), max(0.0, median), 0.1


def _category_options(feature: str, metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("category_options", {}).get(feature, [])
    clean = sorted({str(value).strip() for value in values if str(value).strip()})
    if not clean:
        return ["SIN_DATO"]
    return clean


def _default_category_index(options: list[str]) -> int:
    for index, option in enumerate(options):
        if option != "SIN_DATO":
            return index
    return 0


def _join_multiselect(values: list[str]) -> str:
    return " | ".join(sorted(values)) if values else "SIN_DATO"


def _model_score(model: Any, frame: pd.DataFrame) -> tuple[float | None, str]:
    """Return a ranking score without implying probability calibration."""

    if hasattr(model, "predict_proba"):
        value = float(np.asarray(model.predict_proba(frame))[0, 1])
        return value, "Escala interna de 0 a 1; no calibrada."
    if hasattr(model, "decision_function"):
        value = float(np.asarray(model.decision_function(frame)).reshape(-1)[0])
        return value, "Margen interno del clasificador; no calibrado."
    return None, "El clasificador no expone un puntaje continuo."


def _metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return "—" if value is None else f"{float(value):.3f}"


def _render_sidebar(metadata: dict[str, Any], results: dict[str, Any]) -> None:
    selected = metadata.get("model_selection", {}).get(
        "selected_model", "Modelo serializado"
    )
    with st.sidebar:
        st.markdown("### Proyecto final")
        st.caption("Brayam Arboleda Diaz · Saul Dario Gomez")
        st.markdown("**Modelo desplegado**")
        st.write(selected)
        st.markdown("**Fuente oficial**")
        st.link_button("Abrir Datos Abiertos Colombia", SOURCE_URL, width="stretch")
        snapshot = metadata.get("source", {}).get("snapshot", "2026-09-21")
        st.caption(f"Corte analizado: {snapshot}")
        if results:
            data = results.get("data", {})
            st.divider()
            st.caption(
                f"{int(data.get('modeling_rows', 0)):,} presentaciones CUM · "
                f"{int(data.get('unique_expedientes', 0)):,} expedientes"
            )
        st.divider()
        st.warning(
            "Uso exclusivamente académico. El resultado no reemplaza una decisión "
            "oficial ni una revisión regulatoria de INVIMA.",
            icon="⚠️",
        )


def _prediction_tab(model: Any, metadata: dict[str, Any]) -> None:
    st.subheader("Evaluación de una nueva presentación")
    st.write(
        "Complete atributos estructurales disponibles antes de la clasificación. "
        "No se usan nombres comerciales, descripciones, fechas de inactivación ni "
        "identificadores como predictores."
    )

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("#### Estructura de la presentación")
        numeric_values: dict[str, int | float] = {}
        numeric_columns = st.columns(2)
        for index, feature in enumerate(NUMERIC_FEATURES):
            minimum, maximum, default, step = _numeric_settings(feature, metadata)
            with numeric_columns[index % 2]:
                numeric_values[feature] = st.number_input(
                    LABELS[feature],
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    help=HELP_TEXT.get(feature),
                    key=f"input_{feature}",
                )

        st.markdown("#### Clasificación y administración")
        categorical_values: dict[str, str] = {}
        categorical_columns = st.columns(2)
        for index, feature in enumerate(SINGLE_CATEGORICAL_FEATURES):
            options = _category_options(feature, metadata)
            with categorical_columns[index % 2]:
                categorical_values[feature] = st.selectbox(
                    LABELS[feature],
                    options=options,
                    index=_default_category_index(options),
                    help=HELP_TEXT.get(feature),
                    key=f"input_{feature}",
                )

        st.markdown("#### Atributos con múltiples valores")
        st.caption("Puede elegir más de una opción en cada campo.")
        multi_columns = st.columns(2)
        for index, feature in enumerate(MULTI_CATEGORICAL_FEATURES):
            options = [
                item
                for item in _category_options(feature, metadata)
                if item != "SIN_DATO"
            ]
            with multi_columns[index % 2]:
                selected = st.multiselect(
                    LABELS[feature],
                    options=options,
                    help=HELP_TEXT.get(feature),
                    key=f"input_{feature}",
                )
                categorical_values[feature] = _join_multiselect(selected)

        submitted = st.form_submit_button(
            "Clasificar presentación", type="primary", width="stretch"
        )

    if submitted:
        row = {**numeric_values, **categorical_values}
        frame = pd.DataFrame([row], columns=MODEL_FEATURES)
        prediction = int(np.asarray(model.predict(frame)).reshape(-1)[0])
        score, score_note = _model_score(model, frame)
        st.session_state["last_prediction"] = {
            "prediction": prediction,
            "score": score,
            "score_note": score_note,
            "row": row,
        }

    result = st.session_state.get("last_prediction")
    if not result:
        st.info(
            "Diligencie los campos y seleccione **Clasificar presentación** para "
            "obtener un resultado.",
            icon="ℹ️",
        )
        return

    is_positive = result["prediction"] == 1
    css_class = "cum-result-positive" if is_positive else "cum-result-negative"
    title = "Revisión prioritaria: posible muestra médica" if is_positive else (
        "Sin señal suficiente de muestra médica"
    )
    explanation = (
        "El patrón estructural se parece a presentaciones clasificadas como muestra "
        "médica en el corte estudiado. Conviene revisar la consistencia documental."
        if is_positive
        else "El patrón estructural no activa la clase positiva con el umbral aprendido. "
        "Esto no certifica la clasificación regulatoria."
    )
    st.markdown(
        f'<div class="{css_class}"><h3>{title}</h3><p>{explanation}</p></div>',
        unsafe_allow_html=True,
    )

    score = result.get("score")
    output_columns = st.columns([1, 1, 2])
    output_columns[0].metric("Clase estimada", "Sí" if is_positive else "No")
    output_columns[1].metric(
        "Puntaje de priorización no calibrado",
        "—" if score is None else f"{float(score):.3f}",
        help=result.get("score_note"),
    )
    output_columns[2].info(
        "El puntaje ordena casos según la señal del modelo; no expresa una "
        "probabilidad real ni un nivel de certeza.",
        icon="🧭",
    )

    with st.expander("Ver datos utilizados en esta estimación"):
        readable = pd.DataFrame(
            {
                "Variable": [LABELS[item] for item in MODEL_FEATURES],
                # Streamlit/Arrow requires one consistent dtype per displayed column.
                "Valor": [str(result["row"][item]) for item in MODEL_FEATURES],
            }
        )
        st.dataframe(readable, hide_index=True, width="stretch")
        export = readable.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar registro de la estimación",
            data=export,
            file_name="estimacion_muestra_medica.csv",
            mime="text/csv",
        )


def _methodology_tab(metadata: dict[str, Any], results: dict[str, Any]) -> None:
    st.subheader("Metodología CRISP-DM")
    st.write(
        "La solución sigue las seis fases vistas en clase y conserva el flujo "
        "completo de transformación dentro del pipeline serializado."
    )
    steps = [
        (
            "1 · Comprensión del negocio",
            "Priorizar presentaciones que ameritan revisión por su posible condición "
            "de muestra médica; no sustituir la decisión de INVIMA.",
        ),
        (
            "2 · Comprensión de los datos",
            "Se estudió el CUM vigente de Datos Abiertos Colombia, su diccionario, "
            "calidad, duplicación lógica, faltantes y distribución del objetivo.",
        ),
        (
            "3 · Preparación",
            "Se consolidó una fila por presentación CUM y se excluyeron texto, fechas, "
            "identificadores y campos que filtran directa o indirectamente el objetivo.",
        ),
        (
            "4 · Modelado",
            "Se compararon cinco modelos clásicos y tres ensambles. Imputación, "
            "codificación, escalado y sobremuestreo ocurren dentro de cada fold.",
        ),
        (
            "5 · Evaluación",
            "La selección se cerró con F1 y validación cruzada agrupada antes de usar "
            "el 30 % de test. También se reportan precisión, recall, accuracy y AUC.",
        ),
        (
            "6 · Despliegue",
            "La aplicación carga el mismo pipeline entrenado, valida las 18 variables "
            "y mantiene el puntaje como señal no calibrada de priorización.",
        ),
    ]
    for start in range(0, len(steps), 3):
        columns = st.columns(3)
        for column, (title, body) in zip(columns, steps[start : start + 3]):
            column.markdown(
                f'<div class="cum-step"><strong>{title}</strong><br><span>{body}</span></div>',
                unsafe_allow_html=True,
            )

    holdout = results.get("holdout", metadata.get("holdout", {}))
    st.markdown("#### Diseño de validación")
    validation_columns = st.columns(4)
    validation_columns[0].metric(
        "Entrenamiento",
        f"{float(holdout.get('train_share', 0.70)):.1%}",
    )
    validation_columns[1].metric(
        "Test final",
        f"{float(holdout.get('test_share', 0.30)):.1%}",
    )
    validation_columns[2].metric("Folds internos", "5")
    validation_columns[3].metric(
        "Expedientes compartidos",
        f"{int(holdout.get('group_overlap', 0))}",
    )
    st.caption(
        "La separación se hizo por expediente sanitario, no por fila, para impedir "
        "que presentaciones emparentadas aparezcan a ambos lados de la evaluación."
    )

    selection = metadata.get("model_selection", {})
    with st.expander("Criterio de selección del modelo"):
        st.write(selection.get("reason", "La selección se documenta en los resultados."))


def _performance_tab(results: dict[str, Any]) -> None:
    st.subheader("Desempeño fuera de muestra")
    if not results:
        st.info(
            "Las métricas aparecerán cuando se ejecute el entrenamiento y se generen "
            "los artefactos de evaluación.",
            icon="ℹ️",
        )
        return

    metrics = results.get("final_test_metrics", {})
    columns = st.columns(5)
    for column, (key, label) in zip(
        columns,
        [
            ("accuracy", "Accuracy"),
            ("precision", "Precisión"),
            ("recall", "Recall"),
            ("f1", "F1"),
            ("balanced_accuracy", "Accuracy balanceada"),
        ],
    ):
        column.metric(label, _metric_value(metrics, key))

    st.caption(
        "F1 es la métrica principal porque equilibra falsos positivos —revisiones "
        "innecesarias— y falsos negativos —casos potencialmente omitidos— en una "
        "clase minoritaria. El test se consultó una sola vez tras cerrar la selección."
    )

    plot_columns = st.columns(2)
    if COMPARISON_FIGURE.exists():
        plot_columns[0].image(
            str(COMPARISON_FIGURE),
            caption="F1 en validación cruzada agrupada (media ± desviación estándar)",
            width="stretch",
        )
    if CONFUSION_FIGURE.exists():
        plot_columns[1].image(
            str(CONFUSION_FIGURE),
            caption="Matriz de confusión en el conjunto de test",
            width="stretch",
        )

    with st.expander("Definición de las métricas"):
        st.markdown(
            """
            - **Accuracy:** proporción total de clasificaciones correctas.
            - **Precisión:** entre las alertas positivas, proporción realmente positiva.
            - **Recall:** entre los positivos reales, proporción detectada.
            - **F1:** media armónica entre precisión y recall.
            - **Accuracy balanceada:** promedio del desempeño por clase; evita que la
              clase mayoritaria domine la lectura.
            - **ROC-AUC y PR-AUC:** capacidad de ordenamiento a través de distintos
              umbrales; no convierten el puntaje en una probabilidad calibrada.
            """
        )

    test_metrics = results.get("test_metrics", {})
    if test_metrics:
        table = pd.DataFrame.from_dict(test_metrics, orient="index").reset_index()
        table = table.rename(
            columns={
                "index": "Modelo",
                "accuracy": "Accuracy",
                "precision": "Precisión",
                "recall": "Recall",
                "f1": "F1",
                "balanced_accuracy": "Accuracy balanceada",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
            }
        )
        numeric_columns = table.select_dtypes(include="number").columns
        table[numeric_columns] = table[numeric_columns].round(3)
        with st.expander("Comparación completa en test"):
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption(
                "Esta tabla satisface el reporte comparativo. La elección del modelo "
                "no se modificó después de observar estos resultados."
            )


def _dictionary_tab() -> None:
    st.subheader("Diccionario de variables del modelo")
    descriptions = {
        "cantidad_cum": "Cantidad declarada para la presentación comercial.",
        "numero_filas_fuente": "Filas del archivo original consolidadas en la presentación.",
        "numero_cantidades_ingrediente_distintas": "Valores distintos de cantidad de ingrediente.",
        "numero_principios_activos": "Número de principios activos únicos.",
        "numero_vias_administracion": "Número de vías de administración únicas.",
        "numero_codigos_atc": "Número de códigos ATC únicos.",
        "numero_unidades_medida": "Número de unidades de medida únicas.",
        "numero_unidades_referencia": "Número de unidades de referencia únicas.",
        "numero_tipos_rol": "Número de roles regulatorios únicos.",
        "numero_actores": "Número de establecimientos o actores únicos.",
        "forma_farmaceutica": "Forma farmacéutica registrada.",
        "codigo_concentracion": "Código categórico de concentración del CUM.",
        "unidad_cum": "Unidad declarada para la presentación CUM.",
        "modalidad": "Modalidad del registro sanitario.",
        "vias_administracion": "Conjunto de vías de administración separadas por “ | ”.",
        "unidades_medida": "Conjunto normalizado de unidades de medida.",
        "familias_atc": "Familias terapéuticas derivadas de la primera letra del ATC.",
        "tipos_rol": "Conjunto de tipos de rol asociados a la presentación.",
    }
    rows = []
    for feature in MODEL_FEATURES:
        rows.append(
            {
                "Variable": LABELS[feature],
                "Tipo en la app": (
                    "Numérica" if feature in NUMERIC_FEATURES else (
                        "Selección múltiple"
                        if feature in MULTI_CATEGORICAL_FEATURES
                        else "Categórica"
                    )
                ),
                "Definición": descriptions[feature],
                "Tratamiento": (
                    "Imputación, log(1+x) y estandarización"
                    if feature in NUMERIC_FEATURES
                    else (
                        "Codificación binaria por token"
                        if feature in MULTI_CATEGORICAL_FEATURES
                        else "Imputación y one-hot con categorías desconocidas permitidas"
                    )
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Variable objetivo: `muestramedica` (Sí = 1, No = 0). No se solicita al "
        "usuario porque es el resultado que el modelo estima."
    )


def _limitations_tab(metadata: dict[str, Any], results: dict[str, Any]) -> None:
    st.subheader("Alcance y limitaciones")
    st.markdown(
        """
        - **No es un sistema regulatorio.** Es un prototipo académico para priorizar
          revisiones; la clasificación oficial corresponde a INVIMA.
        - **El puntaje no está calibrado.** El sobremuestreo cambia la prevalencia
          observada durante el ajuste, por lo que el valor continuo sirve para ordenar
          casos, no para afirmar una probabilidad real.
        - **Depende del corte estudiado.** Cambios en la normativa, el catálogo o los
          patrones de registro pueden degradar el desempeño y exigen reentrenamiento.
        - **Las categorías nuevas se toleran, no se aprenden.** El pipeline evita fallar
          ante valores desconocidos, pero una categoría inédita no aporta señal hasta
          que el modelo se actualice con ejemplos suficientes.
        - **Asociación no implica causalidad.** Los patrones del modelo no explican por
          qué una presentación es una muestra médica.
        - **Los conteos deben ser coherentes.** Las cantidades de vías, unidades y roles
          deben corresponder a las selecciones múltiples realizadas por el usuario.
        """
    )

    st.markdown("#### Controles contra fuga de información")
    controls = pd.DataFrame(
        [
            {
                "Riesgo": "Varias filas para la misma presentación",
                "Control": "Consolidación a una fila por expediente + consecutivo CUM.",
            },
            {
                "Riesgo": "Mismo expediente en entrenamiento y test",
                "Control": "Separación y validación cruzada agrupadas por expediente.",
            },
            {
                "Riesgo": "Texto que revela “muestra médica”",
                "Control": "Nombre y descripción comercial excluidos del modelo.",
            },
            {
                "Riesgo": "Variables posteriores al resultado",
                "Control": "Fechas, estado, inactivación e identificadores excluidos.",
            },
            {
                "Riesgo": "Balanceo antes de dividir",
                "Control": "RandomOverSampler aplicado dentro de cada pipeline/fold.",
            },
        ]
    )
    st.dataframe(controls, hide_index=True, width="stretch")

    source = metadata.get("source", {})
    data = results.get("data", {})
    st.markdown("#### Trazabilidad")
    st.write(
        f"Fuente: **{source.get('publisher', 'INVIMA · Datos Abiertos Colombia')}**. "
        f"El modelado consolidó {int(data.get('raw_rows', 0)):,} filas originales en "
        f"{int(data.get('modeling_rows', 0)):,} presentaciones CUM."
        if data
        else "Fuente: **INVIMA · Datos Abiertos Colombia**."
    )
    st.link_button("Consultar el conjunto oficial", SOURCE_URL)


def main() -> None:
    _inject_styles()
    st.markdown(
        """
        <div class="cum-hero">
            <h1>Clasificador de presentaciones CUM</h1>
            <p>Apoyo académico para priorizar la revisión de una posible condición de
            muestra médica a partir de atributos estructurales del registro.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not METADATA_PATH.exists() or not MODEL_PATH.exists():
        st.error(
            "No se encontraron los artefactos entrenados. Ejecute primero el proceso "
            "de entrenamiento para generar `models/pipeline_muestra_medica.joblib` y "
            "`models/metadata_modelo.json`.",
            icon="🚫",
        )
        st.stop()

    try:
        metadata = _read_json_if_available(METADATA_PATH)
        _validate_metadata(metadata)
        results = _read_json_if_available(RESULTS_PATH)
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as error:
        st.error(
            "La aplicación no pudo cargar un artefacto compatible. Vuelva a ejecutar "
            "el entrenamiento y el despliegue con las mismas dependencias.",
            icon="🚫",
        )
        with st.expander("Detalle técnico"):
            st.code(str(error))
        st.stop()

    _render_sidebar(metadata, results)
    st.warning(metadata.get("warning", "Herramienta académica de apoyo."), icon="⚠️")

    prediction, methodology, performance, dictionary, limitations = st.tabs(
        [
            "Predicción",
            "Metodología",
            "Desempeño",
            "Diccionario",
            "Limitaciones",
        ]
    )
    with prediction:
        _prediction_tab(model, metadata)
    with methodology:
        _methodology_tab(metadata, results)
    with performance:
        _performance_tab(results)
    with dictionary:
        _dictionary_tab()
    with limitations:
        _limitations_tab(metadata, results)

    st.divider()
    st.caption(
        "Proyecto final de Aprendizaje de Máquinas · Brayam Arboleda Diaz y "
        "Saul Dario Gomez · Fuente: Datos Abiertos Colombia / INVIMA"
    )


if __name__ == "__main__":
    main()
