"""Auditoría reproducible de calidad y entendimiento del dataset de medicamentos.

El script trabaja siempre sobre el CSV original y genera artefactos trazables para
las fases de Comprensión de los datos y Preparación de los datos de CRISP-DM.
No transforma ni sobrescribe la fuente.

Uso desde la raíz del proyecto::

    python src/data_quality.py

También se puede indicar otra ruta con ``--input``. El reporte de ydata-profiling
usa el conjunto completo por defecto; ``--profile-rows`` permite fijar una muestra
reproducible solo cuando hay restricciones de memoria.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "codigo_unico_medicamentos_vigentes_20260921.csv"
)
DEFAULT_REPORTS = PROJECT_ROOT / "reports"
DEFAULT_FIGURES = PROJECT_ROOT / "figures"

TARGET = "muestramedica"
GROUP_COLUMNS = ["expedientecum", "consecutivocum"]
DATE_COLUMNS = ["fechaexpedicion", "fechavencimiento", "fechaactivo", "fechainactivo"]


# El diccionario parte del significado publicado por INVIMA y añade la función
# analítica que tendrá cada campo en este proyecto. Debe permanecer sincronizado
# con las 29 columnas del archivo fuente.
FIELD_DOCUMENTATION: dict[str, tuple[str, str, str]] = {
    "expediente": (
        "Identificador del expediente del registro sanitario.",
        "Identificador administrativo",
        "Excluir: identificador de alta cardinalidad y equivalente práctico del expediente CUM.",
    ),
    "producto": (
        "Nombre declarado del medicamento o producto.",
        "Texto descriptivo",
        "Excluir: el texto puede contener literalmente 'muestra médica' y filtrar el objetivo.",
    ),
    "titular": (
        "Persona jurídica titular del registro sanitario.",
        "Categoría de alta cardinalidad",
        "Excluir del modelo base para evitar memorizar actores y reducir sesgo por titular.",
    ),
    "registrosanitario": (
        "Código del registro sanitario otorgado por INVIMA.",
        "Identificador",
        "Excluir: identificador de alta cardinalidad sin interpretación causal.",
    ),
    "fechaexpedicion": (
        "Fecha de expedición del registro sanitario.",
        "Fecha administrativa",
        "Excluir: no representa una característica estructural de la presentación y puede inducir deriva temporal.",
    ),
    "fechavencimiento": (
        "Fecha registrada de vencimiento del registro sanitario.",
        "Fecha administrativa",
        "Excluir: faltantes y fechas centinela (por ejemplo, año 3000).",
    ),
    "estadoregistro": (
        "Estado vigente del registro sanitario.",
        "Categoría administrativa",
        "Excluir: el universo ya corresponde a registros vigentes y aporta variación limitada.",
    ),
    "expedientecum": (
        "Identificador del expediente asociado al Código Único de Medicamento (CUM).",
        "Identificador/grupo",
        "Usar solo como grupo de partición; nunca como predictor.",
    ),
    "consecutivocum": (
        "Consecutivo de la presentación dentro del expediente CUM.",
        "Identificador",
        "Usar con expedientecum para consolidar la presentación; excluir como predictor.",
    ),
    "cantidadcum": (
        "Cantidad de unidades contenidas en la presentación comercial.",
        "Numérica estructural",
        "Candidata; imputar con mediana y aplicar log1p dentro del pipeline.",
    ),
    "descripcioncomercial": (
        "Descripción textual del empaque o presentación comercial.",
        "Texto descriptivo",
        "Excluir: puede mencionar explícitamente 'muestra médica' y causar fuga del objetivo.",
    ),
    "estadocum": (
        "Estado activo o inactivo de la presentación CUM.",
        "Categoría posterior/administrativa",
        "Excluir: estado observado después del registro, no disponible de forma fiable en prevalidación.",
    ),
    "fechaactivo": (
        "Fecha desde la cual la presentación figura activa.",
        "Fecha administrativa",
        "Excluir del modelo: información temporal posterior o simultánea al registro.",
    ),
    "fechainactivo": (
        "Fecha en que la presentación pasó a estado inactivo.",
        "Fecha posterior",
        "Excluir estrictamente: fuga temporal y alto porcentaje de faltantes.",
    ),
    "muestramedica": (
        "Indica si la presentación está registrada como muestra médica (Si/No).",
        "Objetivo binario",
        "Variable objetivo; normalizar Si=1 y No=0, nunca incluir entre predictores.",
    ),
    "unidad": (
        "Unidad usada para expresar la cantidad de la presentación CUM.",
        "Categoría estructural",
        "Candidata; normalizar texto e imputar dentro del pipeline.",
    ),
    "atc": (
        "Código de clasificación Anatómica, Terapéutica y Química (ATC).",
        "Categoría clínica",
        "Agregar por CUM y usar el primer nivel ATC y el número de códigos, no el código completo aislado.",
    ),
    "descripcionatc": (
        "Descripción del código ATC.",
        "Texto/categoría clínica",
        "Excluir por redundancia con atc y alta cardinalidad.",
    ),
    "viaadministracion": (
        "Vía o vías de administración del medicamento.",
        "Categoría estructural multivaluada",
        "Agregar como conjunto ordenado por CUM y contar vías.",
    ),
    "concentracion": (
        "Indicador o clasificación de concentración reportada.",
        "Categoría estructural",
        "Candidata; imputar dentro del pipeline.",
    ),
    "principioactivo": (
        "Nombre del ingrediente o principio activo asociado a la presentación.",
        "Categoría multivaluada",
        "No usar el nombre completo por cardinalidad; agregar y usar el número de principios activos.",
    ),
    "unidadmedida": (
        "Unidad de medida de la cantidad del ingrediente.",
        "Categoría estructural multivaluada",
        "Normalizar abreviaturas, agregar como conjunto por CUM y contar unidades distintas.",
    ),
    "cantidad": (
        "Cantidad reportada para cada ingrediente, expresada en unidadmedida.",
        "Numérica por ingrediente",
        "No usar directamente: mezcla escalas/unidades y varias filas por presentación.",
    ),
    "unidadreferencia": (
        "Unidad o presentación de referencia para interpretar la cantidad del ingrediente.",
        "Categoría de alta cardinalidad",
        "Excluir del modelo base por heterogeneidad textual y redundancia con la forma/unidad.",
    ),
    "formafarmaceutica": (
        "Forma farmacéutica de la presentación (tableta, solución, etc.).",
        "Categoría estructural",
        "Candidata; imputar y codificar dentro del pipeline.",
    ),
    "nombrerol": (
        "Nombre de la organización que cumple un rol en el registro.",
        "Categoría de alta cardinalidad",
        "Excluir nombres; usar solo el número de actores distintos por CUM.",
    ),
    "tiporol": (
        "Tipo de rol de la organización (fabricante, importador, etc.).",
        "Categoría multivaluada",
        "Agregar como conjunto por CUM y contar roles distintos.",
    ),
    "modalidad": (
        "Modalidad autorizada del registro (fabricar y vender, importar, etc.).",
        "Categoría estructural",
        "Candidata; imputar y codificar dentro del pipeline.",
    ),
    "IUM": (
        "Identificador Único de Medicamento (IUM), cuando está disponible.",
        "Identificador",
        "Excluir: alta ausencia y carácter identificador.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita el CSV original del proyecto.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument(
        "--profile-rows",
        type=int,
        default=0,
        help="0 usa todas las filas; un entero positivo usa una muestra reproducible.",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Omite ydata-profiling (útil solo para pruebas rápidas del script).",
    )
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    """Convierte escalares numpy/pandas y valores no finitos a JSON seguro."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_target(series: pd.Series) -> pd.Series:
    text = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(lambda x: _strip_accents(x) if pd.notna(x) else x)
    )
    unexpected = sorted(text.dropna().loc[~text.dropna().isin(["si", "no"])].unique())
    if unexpected:
        raise ValueError(f"Valores inesperados en {TARGET}: {unexpected}")
    if text.isna().any():
        raise ValueError(f"{TARGET} contiene {int(text.isna().sum())} valores ausentes.")
    return text.map({"si": 1, "no": 0}).astype("int8")


def read_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo fuente: {path}")
    data = pd.read_csv(path, low_memory=False)
    expected = list(FIELD_DOCUMENTATION)
    missing = sorted(set(expected) - set(data.columns))
    unexpected = sorted(set(data.columns) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "El esquema no coincide con el diccionario esperado. "
            f"Faltantes={missing}; inesperadas={unexpected}"
        )
    return data[expected]


@lru_cache(maxsize=None)
def _parse_one_date(value: str) -> datetime | None:
    """Interpreta fechas incluso si el año queda fuera del rango de pandas.

    La fuente contiene años centinela como 3000. Pandas los convierte en NaT por
    el límite de datetime64[ns], lo cual los confundiría con errores sintácticos.
    ``datetime`` de Python sí permite reconocerlos y luego reportarlos aparte.
    """

    for date_format in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    return None


def _date_parse(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.map(lambda value: _parse_one_date(str(value)) if pd.notna(value) else None)


def _date_year(series: pd.Series) -> pd.Series:
    parsed = _date_parse(series)
    return parsed.map(lambda value: float(value.year) if value is not None else np.nan)


def build_quality_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_rows = len(data)
    for column in data.columns:
        series = data[column]
        non_null = int(series.notna().sum())
        nulls = n_rows - non_null
        uniques = int(series.nunique(dropna=True))
        blank = 0
        examples: list[str] = []
        minimum: Any = None
        maximum: Any = None
        invalid_dates = 0
        outlier_date_count = 0
        notes: list[str] = []

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            text = series.astype("string")
            blank = int(text.str.strip().eq("").fillna(False).sum())
            examples = [str(item)[:120] for item in text.dropna().unique()[:3]]
        else:
            examples = [str(item)[:120] for item in series.dropna().unique()[:3]]

        if column in DATE_COLUMNS:
            parsed = _date_parse(series)
            invalid_dates = int(series.notna().sum() - parsed.notna().sum())
            valid = parsed.dropna()
            if not valid.empty:
                minimum = min(valid).date().isoformat()
                maximum = max(valid).date().isoformat()
                years = valid.map(lambda value: value.year)
                outlier_date_count = int(((years < 1950) | (years > 2100)).sum())
            if outlier_date_count:
                notes.append(f"{outlier_date_count:,} fechas fuera de 1950–2100")
            if invalid_dates:
                notes.append(f"{invalid_dates:,} fechas no interpretables")
        elif pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                minimum = float(numeric.min())
                maximum = float(numeric.max())
                if (numeric < 0).any():
                    notes.append(f"{int((numeric < 0).sum()):,} valores negativos")

        if nulls / max(n_rows, 1) >= 0.5:
            notes.append("ausencia ≥ 50%")
        if uniques / max(non_null, 1) >= 0.95 and uniques > 100:
            notes.append("alta cardinalidad/casi identificador")
        if blank:
            notes.append(f"{blank:,} cadenas vacías")

        rows.append(
            {
                "variable": column,
                "tipo_pandas": str(series.dtype),
                "filas": n_rows,
                "no_nulos": non_null,
                "nulos": nulls,
                "porcentaje_nulos": round(100 * nulls / max(n_rows, 1), 4),
                "valores_unicos": uniques,
                "porcentaje_unicidad_sobre_no_nulos": round(
                    100 * uniques / max(non_null, 1), 4
                ),
                "cadenas_vacias": blank,
                "minimo": minimum,
                "maximo": maximum,
                "fechas_no_interpretables": invalid_dates,
                "fechas_fuera_1950_2100": outlier_date_count,
                "ejemplos": " | ".join(examples),
                "hallazgos": "; ".join(notes) if notes else "Sin alerta automática",
            }
        )
    return pd.DataFrame(rows)


def build_data_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    quality = build_quality_table(data).set_index("variable")
    rows = []
    for column in data.columns:
        description, semantic_type, recommendation = FIELD_DOCUMENTATION[column]
        rows.append(
            {
                "variable": column,
                "descripcion": description,
                "tipo_semantico": semantic_type,
                "tipo_pandas_observado": quality.at[column, "tipo_pandas"],
                "porcentaje_nulos": quality.at[column, "porcentaje_nulos"],
                "valores_unicos": quality.at[column, "valores_unicos"],
                "uso_modelado_y_tratamiento": recommendation,
            }
        )
    dictionary = pd.DataFrame(rows)
    if len(dictionary) != 29:
        raise AssertionError(f"El diccionario debe tener 29 filas; tiene {len(dictionary)}.")
    return dictionary


def presentation_level(data: pd.DataFrame) -> pd.DataFrame:
    """Consolida a una fila por presentación para asociaciones no infladas."""

    grouped = data.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    target_conflicts = grouped[TARGET].nunique(dropna=False)
    if (target_conflicts > 1).any():
        raise ValueError(
            f"Hay {int((target_conflicts > 1).sum())} CUM con objetivo contradictorio."
        )

    representative = grouped.first().reset_index()
    representative["numero_filas_fuente"] = grouped.size().to_numpy()
    representative["numero_principios_activos"] = (
        grouped["principioactivo"].nunique(dropna=True).to_numpy()
    )
    representative["numero_vias_administracion"] = (
        grouped["viaadministracion"].nunique(dropna=True).to_numpy()
    )
    representative["numero_codigos_atc"] = grouped["atc"].nunique(dropna=True).to_numpy()
    representative["numero_roles"] = grouped["tiporol"].nunique(dropna=True).to_numpy()
    representative["numero_actores"] = grouped["nombrerol"].nunique(dropna=True).to_numpy()
    return representative


def _bias_corrected_cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    """V de Cramér corregida por sesgo y p-valor chi-cuadrado."""

    if min(table.shape) < 2 or table.to_numpy().sum() == 0:
        return float("nan"), float("nan")
    chi2, p_value, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, cols = table.shape
    phi2_corrected = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / max(n - 1, 1))
    rows_corrected = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    cols_corrected = cols - ((cols - 1) ** 2) / max(n - 1, 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)
    value = math.sqrt(phi2_corrected / denominator) if denominator > 0 else float("nan")
    return float(value), float(p_value)


def _categorical_for_association(series: pd.Series, max_levels: int = 50) -> pd.Series:
    values = series.astype("string").fillna("<AUSENTE>").str.strip()
    if values.nunique(dropna=False) <= max_levels:
        return values
    keep = set(values.value_counts(dropna=False).head(max_levels - 1).index)
    return values.where(values.isin(keep), "<OTROS>")


def build_associations(presentations: pd.DataFrame) -> pd.DataFrame:
    """Mide asociaciones bivariadas con el objetivo a nivel CUM.

    La tabla es exploratoria, no una selección automática de variables. Para
    variables categóricas de alta cardinalidad se conservan los 49 niveles más
    frecuentes y se agrupa el resto, evitando matrices inmanejables.
    """

    y = normalize_target(presentations[TARGET])
    rows: list[dict[str, Any]] = []
    for column in presentations.columns:
        if column in {TARGET, *GROUP_COLUMNS}:
            continue
        raw = presentations[column]
        note = "Asociación exploratoria; no implica causalidad."

        if column in DATE_COLUMNS:
            numeric = _date_year(raw)
            mask = numeric.notna()
            if mask.sum() > 2 and numeric.loc[mask].nunique() > 1:
                statistic, p_value = pointbiserialr(y.loc[mask], numeric.loc[mask])
            else:
                statistic, p_value = float("nan"), float("nan")
            method = "Correlación punto-biserial (año)"
            variable_type = "fecha"
            n_used = int(mask.sum())
            note += " Fecha excluida del modelo por temporalidad/calidad."
        elif pd.api.types.is_numeric_dtype(raw):
            numeric = pd.to_numeric(raw, errors="coerce")
            mask = numeric.notna()
            if mask.sum() > 2 and numeric.loc[mask].nunique() > 1:
                statistic, p_value = pointbiserialr(y.loc[mask], numeric.loc[mask])
            else:
                statistic, p_value = float("nan"), float("nan")
            method = "Correlación punto-biserial"
            variable_type = "numérica"
            n_used = int(mask.sum())
        else:
            category = _categorical_for_association(raw)
            contingency = pd.crosstab(category, y, dropna=False)
            statistic, p_value = _bias_corrected_cramers_v(contingency)
            method = "V de Cramér corregida"
            variable_type = "categórica"
            n_used = len(category)
            if raw.nunique(dropna=True) > 50:
                note += " Niveles raros agrupados en <OTROS> para el cálculo."

        rows.append(
            {
                "variable": column,
                "tipo": variable_type,
                "metodo": method,
                "asociacion": _json_value(statistic),
                "asociacion_absoluta": _json_value(abs(statistic))
                if math.isfinite(float(statistic))
                else None,
                "p_valor": _json_value(p_value),
                "observaciones_usadas": n_used,
                "valores_unicos_originales": int(raw.nunique(dropna=True)),
                "nota": note,
            }
        )
    associations = pd.DataFrame(rows).sort_values(
        "asociacion_absoluta", ascending=False, na_position="last"
    )
    return associations.reset_index(drop=True)


def _sentinel_date_counts(data: pd.DataFrame) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for column in DATE_COLUMNS:
        parsed = _date_parse(data[column])
        years = _date_year(data[column])
        output[column] = {
            "no_interpretable": int(data[column].notna().sum() - parsed.notna().sum()),
            "anio_menor_1950": int((years < 1950).fillna(False).sum()),
            "anio_mayor_2100": int((years > 2100).fillna(False).sum()),
            "anio_3000": int((years == 3000).fillna(False).sum()),
            "anio_9999": int((years == 9999).fillna(False).sum()),
        }
    return output


def _encoding_issue_counts(data: pd.DataFrame) -> dict[str, int]:
    object_columns = data.select_dtypes(include=["object", "string"]).columns
    replacement = 0
    mojibake = 0
    for column in object_columns:
        text = data[column].astype("string")
        replacement += int(text.str.contains("\ufffd", regex=False, na=False).sum())
        mojibake += int(text.str.contains(r"Ã.|Â.|â€", regex=True, na=False).sum())
    return {
        "celdas_con_caracter_reemplazo": replacement,
        "celdas_con_patron_mojibake": mojibake,
    }


def build_summary(data: pd.DataFrame, presentations: pd.DataFrame, source: Path) -> dict[str, Any]:
    group_sizes = data.groupby(GROUP_COLUMNS, dropna=False).size()
    y_raw = normalize_target(data[TARGET])
    y_cum = normalize_target(presentations[TARGET])
    group_target_nunique = data.groupby(GROUP_COLUMNS, dropna=False)[TARGET].nunique(dropna=False)

    # Diagnóstico reproducible del error que produciría una partición aleatoria por
    # filas: la misma presentación aparecería simultáneamente en train y test.
    rng = np.random.default_rng(42)
    random_test = rng.random(len(data)) < 0.30
    train_groups = set(map(tuple, data.loc[~random_test, GROUP_COLUMNS].to_numpy()))
    test_groups = set(map(tuple, data.loc[random_test, GROUP_COLUMNS].to_numpy()))
    leaked_groups = train_groups & test_groups
    test_row_groups = list(map(tuple, data.loc[random_test, GROUP_COLUMNS].to_numpy()))
    leaked_test_rows = float(
        np.mean([group in train_groups for group in test_row_groups])
    )

    # Incluso dividir por presentación puede dejar expedientes relacionados en
    # ambos subconjuntos; por eso el entrenamiento debe agrupar por expedientecum.
    random_test_cum = rng.random(len(presentations)) < 0.30
    train_expedientes = set(presentations.loc[~random_test_cum, "expedientecum"])
    test_expedientes = presentations.loc[random_test_cum, "expedientecum"]
    leaked_test_cum = float(test_expedientes.isin(train_expedientes).mean())

    # Una simple búsqueda de la frase objetivo en los textos da un desempeño casi
    # perfecto. Esto no es una señal válida: demuestra fuga directa y justifica
    # excluir producto y descripcioncomercial del modelado.
    combined_text = (
        presentations["producto"].astype("string").fillna("")
        + " "
        + presentations["descripcioncomercial"].astype("string").fillna("")
    )
    normalized_text = combined_text.map(lambda value: _strip_accents(str(value)).lower())
    text_prediction = normalized_text.str.contains(r"muestra\s*medica", regex=True).astype("int8")
    true_positive = int(((text_prediction == 1) & (y_cum == 1)).sum())
    false_positive = int(((text_prediction == 1) & (y_cum == 0)).sum())
    false_negative = int(((text_prediction == 0) & (y_cum == 1)).sum())
    text_accuracy = float((text_prediction == y_cum).mean())
    text_precision = true_positive / max(true_positive + false_positive, 1)
    text_recall = true_positive / max(true_positive + false_negative, 1)

    null_percent = (100 * data.isna().mean()).sort_values(ascending=False)
    summary: dict[str, Any] = {
        "fuente": {
            "archivo": f"data/{source.name}",
            "tamano_bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "url_oficial": "https://www.datos.gov.co/Salud-y-Protecci-n-Social/C-DIGO-NICO-DE-MEDICAMENTOS-VIGENTES/i7cb-raxc/about_data",
            "entidad": "Instituto Nacional de Vigilancia de Medicamentos y Alimentos (INVIMA)",
            "portal": "Datos Abiertos Colombia",
            "fecha_corte_archivo": "2026-09-21",
        },
        "dimensiones": {
            "filas_originales": len(data),
            "columnas_originales": data.shape[1],
            "presentaciones_cum_unicas": int(len(group_sizes)),
            "expedientes_cum_unicos": int(data["expedientecum"].nunique(dropna=True)),
            "filas_duplicadas_exactas": int(data.duplicated().sum()),
            "filas_adicionales_por_granularidad_ingrediente_rol_via": int(
                len(data) - len(group_sizes)
            ),
        },
        "granularidad": {
            "clave_presentacion": GROUP_COLUMNS,
            "porcentaje_filas_en_cum_repetidos": float(
                100 * data.set_index(GROUP_COLUMNS).index.duplicated(keep=False).mean()
            ),
            "filas_promedio_por_cum": float(group_sizes.mean()),
            "filas_mediana_por_cum": float(group_sizes.median()),
            "filas_maximas_por_cum": int(group_sizes.max()),
            "cum_con_mas_de_una_fila": int((group_sizes > 1).sum()),
            "cum_con_objetivo_contradictorio": int((group_target_nunique > 1).sum()),
            "advertencia": (
                "La fila original representa combinaciones de ingrediente, rol y vía; "
                "el caso de modelado es la presentación CUM. Se debe consolidar antes de entrenar."
            ),
        },
        "objetivo": {
            "variable": TARGET,
            "conteos_filas_originales": {
                "No": int((y_raw == 0).sum()),
                "Si": int((y_raw == 1).sum()),
            },
            "conteos_presentaciones_cum": {
                "No": int((y_cum == 0).sum()),
                "Si": int((y_cum == 1).sum()),
            },
            "porcentaje_si_presentaciones_cum": float(100 * y_cum.mean()),
            "razon_mayoritaria_sobre_minoritaria": float((y_cum == 0).sum() / (y_cum == 1).sum()),
            "conclusion_balance": (
                "Desbalance moderado (clase positiva cercana al 22%); aplicar remuestreo "
                "solo dentro de cada fold de entrenamiento y priorizar F1."
            ),
        },
        "faltantes": {
            "porcentaje_por_variable": {
                key: float(value) for key, value in null_percent.items()
            },
            "variables_con_50_por_ciento_o_mas": [
                key for key, value in null_percent.items() if value >= 50
            ],
        },
        "fechas": _sentinel_date_counts(data),
        "codificacion_texto": _encoding_issue_counts(data),
        "riesgo_fuga_particion": {
            "semilla_diagnostico": 42,
            "cum_en_ambos_conjuntos_si_se_dividen_filas_aleatoriamente": len(leaked_groups),
            "porcentaje_cum_test_filtrados_a_train_en_split_por_filas": float(
                100 * len(leaked_groups) / max(len(test_groups), 1)
            ),
            "porcentaje_filas_test_cuyo_cum_aparece_en_train": float(100 * leaked_test_rows),
            "porcentaje_presentaciones_test_con_expediente_en_train_si_split_por_cum": float(
                100 * leaked_test_cum
            ),
            "decision": (
                "Usar partición estratificada y agrupada por expedientecum; el test permanece "
                "aislado y las transformaciones/remuestreo se ajustan solo con entrenamiento."
            ),
        },
        "variables_con_fuga_directa_o_identificadores": {
            "texto_que_puede_revelar_objetivo": ["producto", "descripcioncomercial"],
            "informacion_posterior": ["estadocum", "fechaactivo", "fechainactivo", "fechavencimiento"],
            "identificadores": [
                "expediente",
                "expedientecum",
                "consecutivocum",
                "registrosanitario",
                "IUM",
            ],
            "diagnostico_fuga_textual": {
                "regla": "Buscar la frase 'muestra médica' en producto + descripcioncomercial",
                "exactitud": text_accuracy,
                "precision": text_precision,
                "recall": text_recall,
                "conclusion": (
                    "El desempeño artificialmente alto confirma que ambos textos revelan el objetivo; "
                    "se excluyen antes de entrenar."
                ),
            },
        },
    }
    return summary


def make_figures(
    data: pd.DataFrame,
    presentations: pd.DataFrame,
    quality: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # Distribución objetivo al grano correcto.
    counts = (
        presentations[TARGET]
        .astype("string")
        .str.strip()
        .replace({"Si": "Sí"})
        .value_counts()
        .reindex(["No", "Sí"])
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    palette = ["#2E6F9E", "#E67828"]
    bars = ax.bar(counts.index, counts.values, color=palette)
    ax.set_title("Distribución de muestra médica a nivel de presentación CUM")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Presentaciones CUM")
    ax.set_ylim(0, float(counts.max()) * 1.18)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + float(counts.max()) * 0.01,
            f"{int(count):,}\n({100 * count / counts.sum():.1f}%)",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "distribucion_objetivo_cum.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Faltantes por variable.
    missing = quality.sort_values("porcentaje_nulos", ascending=True)
    fig_height = max(6.5, 0.29 * len(missing))
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    colors = ["#C94C4C" if value >= 50 else "#4C8BB8" for value in missing["porcentaje_nulos"]]
    ax.barh(missing["variable"], missing["porcentaje_nulos"], color=colors)
    ax.axvline(50, color="#8B1A1A", linestyle="--", linewidth=1, label="50%")
    ax.set_title("Porcentaje de valores ausentes por variable")
    ax.set_xlabel("Valores ausentes (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "faltantes_por_variable.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Multiplicidad que evidencia el grano original.
    group_sizes = data.groupby(GROUP_COLUMNS, dropna=False).size().clip(upper=20)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bins = np.arange(0.5, 21.6, 1)
    ax.hist(group_sizes, bins=bins, color="#5B8F3A", edgecolor="white")
    ax.set_title("Filas fuente por presentación CUM (20 agrupa valores ≥20)")
    ax.set_xlabel("Número de filas fuente")
    ax.set_ylabel("Presentaciones CUM")
    ax.set_xticks(range(1, 21, 2))
    fig.tight_layout()
    fig.savefig(output_dir / "multiplicidad_filas_por_cum.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_ydata_profile(data: pd.DataFrame, output: Path, profile_rows: int) -> dict[str, Any]:
    try:
        from ydata_profiling import ProfileReport
    except ImportError as exc:  # pragma: no cover - mensaje operativo
        raise RuntimeError(
            "Falta ydata-profiling. Instale las dependencias del proyecto antes de ejecutar."
        ) from exc

    if profile_rows > 0 and profile_rows < len(data):
        profiled = data.sample(n=profile_rows, random_state=42).reset_index(drop=True)
        sampling = {
            "filas_perfiladas": profile_rows,
            "filas_fuente": len(data),
            "muestra": True,
            "semilla": 42,
        }
    else:
        profiled = data
        sampling = {
            "filas_perfiladas": len(data),
            "filas_fuente": len(data),
            "muestra": False,
            "semilla": None,
        }

    profile = ProfileReport(
        profiled,
        title="Perfil de datos — Código Único de Medicamentos Vigentes",
        minimal=True,
        explorative=False,
        progress_bar=True,
    )
    profile.to_file(output)
    return sampling


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    reports_dir = args.reports_dir.expanduser().resolve()
    figures_dir = args.figures_dir.expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo fuente sin modificar: {source}")
    data = read_source(source)
    print(f"Fuente leída: {len(data):,} filas × {data.shape[1]} columnas")

    quality = build_quality_table(data)
    dictionary = build_data_dictionary(data)
    presentations = presentation_level(data)
    associations = build_associations(presentations)
    summary = build_summary(data, presentations, source)

    quality_path = reports_dir / "calidad_datos.csv"
    dictionary_path = reports_dir / "diccionario_datos_original.csv"
    associations_path = reports_dir / "asociaciones_variables.csv"
    summary_path = reports_dir / "resumen_calidad.json"
    profile_path = reports_dir / "reporte_ydata_profiling.html"

    quality.to_csv(quality_path, index=False, encoding="utf-8-sig")
    dictionary.to_csv(dictionary_path, index=False, encoding="utf-8-sig")
    associations.to_csv(associations_path, index=False, encoding="utf-8-sig")
    make_figures(data, presentations, quality, figures_dir)

    if args.skip_profile:
        summary["perfil_ydata"] = {
            "generado": False,
            "motivo": "Ejecución solicitada con --skip-profile",
        }
    else:
        print("Generando reporte ydata-profiling (modo mínimo)...")
        sampling = build_ydata_profile(data, profile_path, args.profile_rows)
        summary["perfil_ydata"] = {
            "generado": True,
            "archivo": f"reports/{profile_path.name}",
            **sampling,
        }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=_json_value)

    print("Artefactos generados:")
    for path in [quality_path, dictionary_path, associations_path, summary_path]:
        print(f"- {path}")
    if profile_path.exists():
        print(f"- {profile_path}")
    for figure in sorted(figures_dir.glob("*.png")):
        print(f"- {figure}")


if __name__ == "__main__":
    main()
