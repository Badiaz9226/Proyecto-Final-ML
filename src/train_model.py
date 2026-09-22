from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import f_oneway, friedmanchisquare, ttest_rel
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.transformers import split_pipe_tokens


RANDOM_STATE = 42
TARGET_COLUMN = "es_muestra_medica"
GROUP_COLUMN = "expedientecum"
KEY_COLUMNS = ["expedientecum", "consecutivocum"]

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


def _strip_accents(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )


def _normalize_target(value: Any) -> int:
    text = _strip_accents(value).strip().casefold()
    if text not in {"si", "no"}:
        raise ValueError(f"Valor no valido en muestramedica: {value!r}")
    return int(text == "si")


def _combine_values(series: pd.Series) -> str:
    values = {
        re.sub(r"\s+", " ", str(value).strip())
        for value in series.dropna()
        if str(value).strip() and str(value).strip().casefold() != "nan"
    }
    return " | ".join(sorted(values)) if values else "SIN_DATO"


def _normalize_unit(value: Any) -> str:
    if pd.isna(value):
        return "SIN_DATO"
    text = _strip_accents(value).strip().casefold()
    text = text.replace("µ", "u").replace("μ", "u").replace("æ", "u")
    text = re.sub(r"\s+", "", text).rstrip(".")
    aliases = {
        "mg": "mg",
        "g": "g",
        "mcg": "mcg",
        "ug": "mcg",
        "ml": "mL",
        "l": "L",
        "ui": "UI",
        "u.i": "UI",
        "u.i.": "UI",
        "iu": "UI",
        "-": "SIN_DATO",
    }
    return aliases.get(text, text if text else "SIN_DATO")


def _combine_normalized_units(series: pd.Series) -> str:
    values = {_normalize_unit(value) for value in series}
    return " | ".join(sorted(values)) if values else "SIN_DATO"


def _combine_atc_families(series: pd.Series) -> str:
    families = {
        str(value).strip()[0].upper()
        for value in series.dropna()
        if str(value).strip() and str(value).strip()[0].isalpha()
    }
    return " | ".join(sorted(families)) if families else "SIN_DATO"


def _safe_first(series: pd.Series) -> Any:
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else np.nan


def build_modeling_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the exploded source to one row per CUM presentation.

    The raw file repeats a presentation for combinations of active ingredients,
    establishment roles and administration routes. Aggregation prevents the same
    presentation from appearing multiple times in the analytical population.
    """

    required = {
        *KEY_COLUMNS,
        "muestramedica",
        "cantidadcum",
        "formafarmaceutica",
        "concentracion",
        "unidad",
        "modalidad",
        "viaadministracion",
        "unidadmedida",
        "atc",
        "cantidad",
        "principioactivo",
        "unidadreferencia",
        "tiporol",
        "nombrerol",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    if raw[KEY_COLUMNS].isna().any(axis=None):
        affected = int(raw[KEY_COLUMNS].isna().any(axis=1).sum())
        raise ValueError(f"Hay {affected} filas sin clave completa de presentacion CUM.")

    # Fail loudly on target drift instead of silently mapping unexpected values to No.
    raw["muestramedica"].map(_normalize_target)
    grouped = raw.groupby(KEY_COLUMNS, sort=False, dropna=False)
    invariant_columns = [
        "muestramedica",
        "cantidadcum",
        "formafarmaceutica",
        "concentracion",
        "unidad",
        "modalidad",
    ]
    conflicts = {
        column: int((grouped[column].nunique(dropna=False) > 1).sum())
        for column in invariant_columns
    }
    conflicts = {column: count for column, count in conflicts.items() if count}
    if conflicts:
        raise ValueError(f"Atributos inconsistentes dentro de una presentacion: {conflicts}")

    modeling = grouped.agg(
        muestramedica=("muestramedica", _safe_first),
        cantidad_cum=("cantidadcum", _safe_first),
        forma_farmaceutica=("formafarmaceutica", _safe_first),
        codigo_concentracion=("concentracion", _safe_first),
        unidad_cum=("unidad", _safe_first),
        modalidad=("modalidad", _safe_first),
        vias_administracion=("viaadministracion", _combine_values),
        unidades_medida=("unidadmedida", _combine_normalized_units),
        familias_atc=("atc", _combine_atc_families),
        numero_filas_fuente=("muestramedica", "size"),
        numero_cantidades_ingrediente_distintas=("cantidad", "nunique"),
        numero_principios_activos=("principioactivo", "nunique"),
        tipos_rol=("tiporol", _combine_values),
        numero_vias_administracion=("viaadministracion", "nunique"),
        numero_codigos_atc=("atc", "nunique"),
        numero_unidades_medida=("unidadmedida", "nunique"),
        numero_unidades_referencia=("unidadreferencia", "nunique"),
        numero_tipos_rol=("tiporol", "nunique"),
        numero_actores=("nombrerol", "nunique"),
    ).reset_index()

    modeling[TARGET_COLUMN] = modeling["muestramedica"].map(_normalize_target)
    modeling = modeling.drop(columns=["muestramedica"])

    for column in NUMERIC_FEATURES:
        modeling[column] = pd.to_numeric(modeling[column], errors="coerce")
        modeling.loc[modeling[column] < 0, column] = np.nan

    for column in CATEGORICAL_FEATURES:
        modeling[column] = (
            modeling[column]
            .fillna("SIN_DATO")
            .astype(str)
            .str.strip()
            .replace("", "SIN_DATO")
        )

    return modeling


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log1p",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                    validate=False,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )
    multi_value_transformers = [
        (
            f"multi_{column}",
            CountVectorizer(
                tokenizer=split_pipe_tokens,
                token_pattern=None,
                lowercase=False,
                binary=True,
                min_df=10,
            ),
            column,
        )
        for column in MULTI_CATEGORICAL_FEATURES
    ]
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, SINGLE_CATEGORICAL_FEATURES),
            *multi_value_transformers,
        ],
        remainder="drop",
    )


def make_pipeline(estimator: Any) -> ImbPipeline:
    steps: list[tuple[str, Any]] = [("preprocess", make_preprocessor())]
    if isinstance(estimator, KNeighborsClassifier):
        # KNN over the original sparse one-hot space makes exact distances both
        # noisy and prohibitively expensive. SVD is learned inside every fold.
        steps.append(
            (
                "reduce_for_knn",
                TruncatedSVD(n_components=30, random_state=RANDOM_STATE),
            )
        )
    steps.extend(
        [
            ("balance", RandomOverSampler(random_state=RANDOM_STATE)),
            ("model", estimator),
        ]
    )
    return ImbPipeline(
        steps=steps
    )


def model_catalog() -> dict[str, Any]:
    logistic = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="liblinear",
        max_iter=2_000,
        random_state=RANDOM_STATE,
    )
    linear_svm = LinearSVC(
        C=1.0,
        loss="squared_hinge",
        tol=1e-3,
        max_iter=100_000,
        random_state=RANDOM_STATE,
    )
    decision_tree = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )
    neural_network = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        random_state=RANDOM_STATE,
    )
    knn = KNeighborsClassifier(n_neighbors=21, weights="distance", n_jobs=-1)

    voting = VotingClassifier(
        estimators=[
            ("logistic", clone(logistic)),
            ("tree", clone(decision_tree)),
            ("mlp", clone(neural_network)),
        ],
        voting="soft",
        n_jobs=1,
    )
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
        ),
        n_estimators=40,
        max_samples=0.8,
        max_features=0.9,
        bootstrap=True,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    boosting = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),
        n_estimators=100,
        learning_rate=0.5,
        random_state=RANDOM_STATE,
    )

    return {
        "Regresion logistica": logistic,
        "SVM lineal": linear_svm,
        "Red neuronal MLP": neural_network,
        "Arbol de decision": decision_tree,
        "K vecinos mas cercanos": knn,
        "Votacion": voting,
        "Bagging": bagging,
        "Boosting AdaBoost": boosting,
    }


def make_holdout_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic 70/30 group-isolated stratified holdout.

    Ten stratified group folds are assigned first. Three folds form the test
    set and seven form training, keeping every expediente in only one side.
    """

    fold_ids = np.full(len(X), -1, dtype=int)
    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    for fold_id, (_, validation_index) in enumerate(splitter.split(X, y, groups)):
        fold_ids[validation_index] = fold_id
    if (fold_ids < 0).any():
        raise RuntimeError("No fue posible asignar todos los registros a un fold.")
    test_mask = fold_ids < 3
    train_mask = ~test_mask
    return train_mask, test_mask, fold_ids


def _score_values(pipeline: ImbPipeline, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X)
    return None


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    score_values: np.ndarray | None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": None,
        "pr_auc": None,
    }
    if score_values is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, score_values))
        metrics["pr_auc"] = float(average_precision_score(y_true, score_values))
    return metrics


def _serializable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return [_serializable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if pd.isna(value) if not isinstance(value, str) else False:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _split_tokens(values: Iterable[str]) -> list[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(part.strip() for part in str(value).split("|") if part.strip())
    return sorted(tokens)


def build_metadata(
    X_train: pd.DataFrame,
    results: dict[str, Any],
) -> dict[str, Any]:
    numeric_ranges: dict[str, dict[str, float]] = {}
    for column in NUMERIC_FEATURES:
        series = pd.to_numeric(X_train[column], errors="coerce")
        numeric_ranges[column] = {
            "min": float(series.min()),
            "p01": float(series.quantile(0.01)),
            "median": float(series.median()),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
        }

    categories: dict[str, list[str]] = {}
    multi_columns = {
        "vias_administracion",
        "unidades_medida",
        "familias_atc",
        "tipos_rol",
    }
    for column in CATEGORICAL_FEATURES:
        if column in multi_columns:
            categories[column] = _split_tokens(X_train[column].astype(str))
        else:
            categories[column] = sorted(X_train[column].dropna().astype(str).unique())

    return {
        "project_title": "Clasificacion de presentaciones CUM como muestra medica",
        "target": TARGET_COLUMN,
        "positive_class": "Si",
        "negative_class": "No",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_ranges": numeric_ranges,
        "category_options": categories,
        "model_selection": results["model_selection"],
        "holdout": results["holdout"],
        "source": {
            "name": "Codigo Unico de Medicamentos Vigentes",
            "publisher": "INVIMA - Datos Abiertos Colombia",
            "url": "https://www.datos.gov.co/Salud-y-Protecci-n-Social/C-DIGO-NICO-DE-MEDICAMENTOS-VIGENTES/i7cb-raxc/about_data",
            "snapshot": "2026-09-21",
            "sha256": "43cd984ded90ddb8b0bd3d6abdb60a0f7628cd8975cab08fdfb6b61c09e18e6a",
        },
        "score_note": (
            "El valor continuo es un puntaje de priorizacion no calibrado; no es una "
            "probabilidad poblacional porque el entrenamiento usa RandomOverSampler."
        ),
        "warning": (
            "Herramienta academica de apoyo. La prediccion no reemplaza la "
            "clasificacion oficial ni una revision regulatoria de INVIMA."
        ),
    }


@dataclass
class TrainingArtifacts:
    evaluation_pipeline: ImbPipeline
    deployment_pipeline: ImbPipeline
    modeling_table: pd.DataFrame
    results: dict[str, Any]
    test_predictions: pd.DataFrame
    cv_fold_scores: pd.DataFrame
    logistic_grid_results: pd.DataFrame
    svm_grid_results: pd.DataFrame


def train_and_evaluate(raw: pd.DataFrame) -> TrainingArtifacts:
    modeling = build_modeling_table(raw)
    X = modeling[MODEL_FEATURES].copy()
    y = modeling[TARGET_COLUMN].astype(int)
    groups = modeling[GROUP_COLUMN].astype(str)

    train_mask, test_mask, fold_ids = make_holdout_split(X, y, groups)
    X_train, X_test = X.loc[train_mask].copy(), X.loc[test_mask].copy()
    y_train, y_test = y.loc[train_mask].copy(), y.loc[test_mask].copy()
    groups_train = groups.loc[train_mask].copy()

    overlap = set(groups.loc[train_mask]).intersection(set(groups.loc[test_mask]))
    if overlap:
        raise RuntimeError("Se detecto fuga de expedientes entre train y test.")

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_splits = list(cv.split(X_train, y_train, groups_train))
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    catalog = model_catalog()
    cv_rows: list[dict[str, Any]] = []
    fold_f1: dict[str, list[float]] = {}
    cv_summary: dict[str, Any] = {}

    for model_name, estimator in catalog.items():
        pipeline = make_pipeline(estimator)
        validation = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_splits,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            error_score="raise",
        )
        fold_f1[model_name] = validation["test_f1"].tolist()
        cv_summary[model_name] = {}
        for metric in scoring:
            test_values = validation[f"test_{metric}"]
            train_values = validation[f"train_{metric}"]
            cv_summary[model_name][metric] = {
                "mean": float(np.mean(test_values)),
                "std": float(np.std(test_values, ddof=1)),
                "train_mean": float(np.mean(train_values)),
            }
        for fold_number in range(len(cv_splits)):
            cv_rows.append(
                {
                    "modelo": model_name,
                    "fold": fold_number + 1,
                    **{
                        metric: float(validation[f"test_{metric}"][fold_number])
                        for metric in scoring
                    },
                    "train_f1": float(validation["train_f1"][fold_number]),
                }
            )

    # Hyperparameter tuning mirrors the two course models emphasized in class.
    logistic_grid = GridSearchCV(
        estimator=make_pipeline(
            LogisticRegression(
                solver="liblinear",
                max_iter=2_000,
                random_state=RANDOM_STATE,
            )
        ),
        param_grid={
            "model__C": [0.1, 1.0, 10.0],
            # scikit-learn >=1.8 expresses pure L2/L1 through l1_ratio 0/1.
            "model__l1_ratio": [0.0, 1.0],
        },
        scoring="f1",
        cv=cv_splits,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )
    logistic_grid.fit(X_train, y_train, groups=groups_train)

    svm_grid = GridSearchCV(
        estimator=make_pipeline(
            LinearSVC(tol=1e-3, max_iter=100_000, random_state=RANDOM_STATE)
        ),
        param_grid={
            "model__C": [0.1, 1.0, 10.0],
            "model__loss": ["hinge", "squared_hinge"],
        },
        scoring="f1",
        cv=cv_splits,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )
    svm_grid.fit(X_train, y_train, groups=groups_train)

    tuned_candidates = {
        "Regresion logistica ajustada": logistic_grid.best_estimator_,
        "SVM lineal ajustada": svm_grid.best_estimator_,
    }
    tuned_cv: dict[str, Any] = {}
    for model_name, pipeline in tuned_candidates.items():
        validation = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_splits,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            error_score="raise",
        )
        fold_f1[model_name] = validation["test_f1"].tolist()
        tuned_cv[model_name] = {
            metric: {
                "mean": float(np.mean(validation[f"test_{metric}"])),
                "std": float(np.std(validation[f"test_{metric}"], ddof=1)),
                "train_mean": float(np.mean(validation[f"train_{metric}"])),
            }
            for metric in scoring
        }
        for fold_number in range(len(cv_splits)):
            cv_rows.append(
                {
                    "modelo": model_name,
                    "fold": fold_number + 1,
                    **{
                        metric: float(validation[f"test_{metric}"][fold_number])
                        for metric in scoring
                    },
                    "train_f1": float(validation["train_f1"][fold_number]),
                }
            )

    # Statistical comparison uses the aligned group folds.
    candidate_names = list(fold_f1)
    anova = f_oneway(*(fold_f1[name] for name in candidate_names))
    friedman = friedmanchisquare(*(fold_f1[name] for name in candidate_names))

    all_cv_f1_means = {
        **{name: cv_summary[name]["f1"]["mean"] for name in catalog},
        **{name: tuned_cv[name]["f1"]["mean"] for name in tuned_candidates},
    }
    reference_name = "Regresion logistica ajustada"
    logistic_scores = np.array(fold_f1[reference_name])
    paired_comparisons: list[dict[str, Any]] = []
    for name in candidate_names:
        if name == reference_name:
            continue
        scores = np.array(fold_f1[name])
        paired_test = ttest_rel(scores, logistic_scores)
        paired_comparisons.append(
            {
                "candidate": name,
                "mean_difference_vs_tuned_logistic": float(
                    all_cv_f1_means[name] - all_cv_f1_means[reference_name]
                ),
                "t_statistic": float(paired_test.statistic),
                "raw_p": float(paired_test.pvalue),
            }
        )

    ordered = sorted(
        range(len(paired_comparisons)),
        key=lambda index: paired_comparisons[index]["raw_p"],
    )
    running_adjusted = 0.0
    for rank, index in enumerate(ordered):
        raw_p = paired_comparisons[index]["raw_p"]
        adjusted = min(1.0, (len(ordered) - rank) * raw_p)
        running_adjusted = max(running_adjusted, adjusted)
        paired_comparisons[index]["holm_adjusted_p"] = running_adjusted

    eligible = [
        row["candidate"]
        for row in paired_comparisons
        if row["mean_difference_vs_tuned_logistic"] > 0.02
        and row["holm_adjusted_p"] < 0.05
    ]
    if friedman.pvalue < 0.05 and eligible:
        selected_name = max(eligible, key=all_cv_f1_means.get)
    else:
        selected_name = reference_name
    best_name = max(all_cv_f1_means, key=all_cv_f1_means.get)
    f1_gap = float(all_cv_f1_means[best_name] - all_cv_f1_means[reference_name])
    if selected_name in tuned_candidates:
        final_pipeline = clone(tuned_candidates[selected_name])
    else:
        final_pipeline = make_pipeline(clone(catalog[selected_name]))

    if selected_name == reference_name:
        selection_reason = (
            "Se selecciono la regresion logistica ajustada: ningun candidato mostro "
            "simultaneamente una mejora mayor a 0.02 de F1 y evidencia pareada "
            "significativa tras correccion de Holm. Se prefirio la alternativa mas "
            "simple e interpretable. Sus puntajes no se interpretan como probabilidades "
            "calibradas debido al sobremuestreo."
        )
    else:
        selection_reason = (
            f"Se selecciono {selected_name}: frente a la regresion logistica ajustada "
            "mostro una mejora de F1 mayor a 0.02 y evidencia pareada significativa "
            "tras correccion de Holm. La decision se tomo antes de consultar el test."
        )

    # Lock selection before touching the holdout, then report all requested
    # metrics once on the untouched 30% test partition.
    majority_predictions = np.zeros(len(y_test), dtype=int)
    majority_scores = np.full(len(y_test), float(y_train.mean()))
    test_metrics: dict[str, Any] = {
        "Linea base mayoritaria": evaluate_predictions(
            y_test,
            majority_predictions,
            majority_scores,
        )
    }
    test_fitted_models: dict[str, ImbPipeline] = {}
    evaluation_catalog: dict[str, ImbPipeline] = {
        **{name: make_pipeline(clone(estimator)) for name, estimator in catalog.items()},
        **{name: clone(pipeline) for name, pipeline in tuned_candidates.items()},
    }
    for model_name, pipeline in evaluation_catalog.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        scores = _score_values(pipeline, X_test)
        test_metrics[model_name] = evaluate_predictions(y_test, predictions, scores)
        test_fitted_models[model_name] = pipeline

    evaluation_pipeline = clone(final_pipeline).fit(X_train, y_train)
    final_predictions = evaluation_pipeline.predict(X_test)
    final_scores = _score_values(evaluation_pipeline, X_test)
    final_metrics = evaluate_predictions(y_test, final_predictions, final_scores)
    final_confusion = confusion_matrix(y_test, final_predictions).tolist()

    test_rows = modeling.loc[test_mask, KEY_COLUMNS + MODEL_FEATURES + [TARGET_COLUMN]].copy()
    test_rows["prediccion"] = final_predictions
    if final_scores is None:
        test_rows["puntaje_modelo_no_calibrado"] = np.nan
    else:
        test_rows["puntaje_modelo_no_calibrado"] = final_scores
    test_examples = test_rows.head(5).copy()

    # Refit for deployment only after model choice and holdout results are frozen.
    deployment_pipeline = clone(final_pipeline).fit(X, y)

    results: dict[str, Any] = {
        "data": {
            "raw_rows": int(len(raw)),
            "raw_columns": int(raw.shape[1]),
            "modeling_rows": int(len(modeling)),
            "modeling_features": int(len(MODEL_FEATURES)),
            "unique_expedientes": int(modeling[GROUP_COLUMN].nunique()),
            "duplicate_expansion_rows": int(len(raw) - len(modeling)),
            "target_positive_count": int(y.sum()),
            "target_negative_count": int((1 - y).sum()),
            "target_positive_rate": float(y.mean()),
            "majority_accuracy_baseline": float(max(y.mean(), 1 - y.mean())),
            "pr_auc_random_baseline": float(y.mean()),
        },
        "holdout": {
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_share": float(train_mask.mean()),
            "test_share": float(test_mask.mean()),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "train_groups": int(groups.loc[train_mask].nunique()),
            "test_groups": int(groups.loc[test_mask].nunique()),
            "group_overlap": int(len(overlap)),
            "fold_assignments": {
                str(fold): int((fold_ids == fold).sum()) for fold in range(10)
            },
        },
        "cv_summary": cv_summary,
        "tuned_cv_summary": tuned_cv,
        "hyperparameter_tuning": {
            "Regresion logistica": {
                "best_params": logistic_grid.best_params_,
                "best_penalty_alias": (
                    "l1"
                    if logistic_grid.best_params_["model__l1_ratio"] == 1.0
                    else "l2"
                ),
                "best_cv_f1": float(logistic_grid.best_score_),
                "candidates": int(len(logistic_grid.cv_results_["params"])),
            },
            "SVM lineal": {
                "best_params": svm_grid.best_params_,
                "best_cv_f1": float(svm_grid.best_score_),
                "candidates": int(len(svm_grid.cv_results_["params"])),
            },
        },
        "statistical_comparison": {
            "anova_f": float(anova.statistic),
            "anova_p": float(anova.pvalue),
            "friedman_statistic": float(friedman.statistic),
            "friedman_p": float(friedman.pvalue),
            "best_cv_model": best_name,
            "best_cv_f1": float(all_cv_f1_means[best_name]),
            "tuned_logistic_cv_f1": float(
                all_cv_f1_means["Regresion logistica ajustada"]
            ),
            "f1_gap_vs_logistic": f1_gap,
            "paired_comparisons_vs_tuned_logistic": paired_comparisons,
            "practical_f1_margin": 0.02,
            "inference_limitation": (
                "Solo hay cinco folds correlacionados y el ajuste usa esos mismos folds; "
                "los p-valores son evidencia auxiliar, no una prueba definitiva."
            ),
        },
        "model_selection": {
            "selected_model": selected_name,
            "reason": selection_reason,
            "selection_used_test": False,
            "deployment_refit_on_full_dataset_after_evaluation": True,
        },
        "test_metrics": test_metrics,
        "final_test_metrics": final_metrics,
        "final_confusion_matrix": final_confusion,
    }

    return TrainingArtifacts(
        evaluation_pipeline=evaluation_pipeline,
        deployment_pipeline=deployment_pipeline,
        modeling_table=modeling,
        results=results,
        test_predictions=test_examples,
        cv_fold_scores=pd.DataFrame(cv_rows),
        logistic_grid_results=pd.DataFrame(logistic_grid.cv_results_),
        svm_grid_results=pd.DataFrame(svm_grid.cv_results_),
    )


def create_plots(
    artifacts: TrainingArtifacts,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    colors = ["#1F4E79", "#D97A1E"]

    modeling = artifacts.modeling_table
    counts = modeling[TARGET_COLUMN].value_counts().sort_index()
    labels = ["No", "Si"]
    plt.figure(figsize=(7.2, 4.6))
    plot_data = pd.DataFrame(
        {
            "Muestra medica": labels,
            "Presentaciones CUM": [counts.get(0, 0), counts.get(1, 0)],
        }
    )
    ax = sns.barplot(
        data=plot_data,
        x="Muestra medica",
        y="Presentaciones CUM",
        hue="Muestra medica",
        palette=dict(zip(labels, colors)),
        legend=False,
    )
    ax.set_title("Distribucion de la variable objetivo")
    ax.set_xlabel("Muestra medica")
    ax.set_ylabel("Presentaciones CUM")
    for container in ax.containers:
        ax.bar_label(container, fmt="{:,.0f}")
    plt.tight_layout()
    plt.savefig(output_dir / "distribucion_objetivo.png", dpi=180)
    plt.close()

    summaries = {
        **artifacts.results["cv_summary"],
        **artifacts.results["tuned_cv_summary"],
    }
    comparison = pd.DataFrame(
        [
            {
                "modelo": name,
                "f1_mean": metrics["f1"]["mean"],
                "f1_std": metrics["f1"]["std"],
            }
            for name, metrics in summaries.items()
        ]
    ).sort_values("f1_mean")
    plt.figure(figsize=(9.2, 6.0))
    plt.errorbar(
        comparison["f1_mean"],
        comparison["modelo"],
        xerr=comparison["f1_std"],
        fmt="o",
        color="#1F4E79",
        ecolor="#8CA6BF",
        capsize=4,
    )
    plt.title("F1 en validacion cruzada agrupada sobre entrenamiento")
    plt.xlabel("F1 promedio (media +/- desviacion estandar)")
    plt.ylabel("")
    plt.xlim(left=max(0, comparison["f1_mean"].min() - 0.08), right=1)
    plt.tight_layout()
    plt.savefig(output_dir / "comparacion_modelos_cv.png", dpi=180)
    plt.close()

    matrix = np.array(artifacts.results["final_confusion_matrix"])
    plt.figure(figsize=(5.6, 4.8))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("Matriz de confusion en test")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")
    plt.tight_layout()
    plt.savefig(output_dir / "matriz_confusion_test.png", dpi=180)
    plt.close()

    # Recreate the holdout indices deterministically for the ROC curve.
    X = modeling[MODEL_FEATURES]
    y = modeling[TARGET_COLUMN].astype(int)
    groups = modeling[GROUP_COLUMN].astype(str)
    train_mask, test_mask, _ = make_holdout_split(X, y, groups)
    scores = _score_values(artifacts.evaluation_pipeline, X.loc[test_mask])
    if scores is not None:
        fpr, tpr, _ = roc_curve(y.loc[test_mask], scores)
        auc = roc_auc_score(y.loc[test_mask], scores)
        plt.figure(figsize=(6.0, 5.0))
        plt.plot(fpr, tpr, color="#1F4E79", linewidth=2, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="#777777", label="Azar")
        plt.title("Curva ROC del modelo final en test")
        plt.xlabel("Tasa de falsos positivos")
        plt.ylabel("Tasa de verdaderos positivos")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "curva_roc_test.png", dpi=180)
        plt.close()

        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(y.loc[test_mask], scores)
        average_precision = average_precision_score(y.loc[test_mask], scores)
        plt.figure(figsize=(6.0, 5.0))
        plt.plot(
            recall,
            precision,
            color="#D97A1E",
            linewidth=2,
            label=f"AP = {average_precision:.3f}",
        )
        plt.axhline(
            y.loc[test_mask].mean(),
            linestyle="--",
            color="#777777",
            label="Prevalencia",
        )
        plt.title("Curva Precision-Recall del modelo final en test")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(output_dir / "curva_precision_recall_test.png", dpi=180)
        plt.close()

    try:
        model = artifacts.evaluation_pipeline.named_steps["model"]
        if isinstance(model, LogisticRegression):
            names = artifacts.evaluation_pipeline.named_steps[
                "preprocess"
            ].get_feature_names_out()
            coefficients = pd.DataFrame(
                {"variable": names, "coeficiente": model.coef_[0]}
            )
            coefficients["magnitud"] = coefficients["coeficiente"].abs()
            top = coefficients.nlargest(20, "magnitud").sort_values("coeficiente")
            top["direccion"] = np.where(
                top["coeficiente"] >= 0, "Aumenta", "Reduce"
            )
            plt.figure(figsize=(9.2, 7.2))
            sns.barplot(
                data=top,
                x="coeficiente",
                y="variable",
                hue="direccion",
                palette={"Aumenta": "#D97A1E", "Reduce": "#1F4E79"},
                dodge=False,
            )
            plt.title("Variables con mayor peso en el modelo final")
            plt.xlabel("Peso del modelo logistico (asociacion, no causalidad)")
            plt.ylabel("")
            plt.legend(title="Direccion de la asociacion")
            plt.tight_layout()
            plt.savefig(output_dir / "coeficientes_modelo_final.png", dpi=180)
            plt.close()
    except Exception as error:  # Plot is useful but not a training blocker.
        print(f"No se pudo crear el grafico de coeficientes: {error}")


def save_artifacts(
    artifacts: TrainingArtifacts,
    output_root: Path,
) -> None:
    data_dir = output_root / "data"
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"
    for directory in (data_dir, models_dir, reports_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    artifacts.modeling_table.to_csv(
        data_dir / "dataset_modelado_cum.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(
        artifacts.deployment_pipeline,
        models_dir / "pipeline_muestra_medica.joblib",
    )
    _write_json(reports_dir / "resultados_modelado.json", artifacts.results)
    artifacts.test_predictions.to_csv(
        reports_dir / "predicciones_datos_no_vistos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cv_rows: list[dict[str, Any]] = []
    for model_name, metrics in {
        **artifacts.results["cv_summary"],
        **artifacts.results["tuned_cv_summary"],
    }.items():
        cv_rows.append(
            {
                "modelo": model_name,
                **{
                    f"{metric}_{stat}": values[stat]
                    for metric, values in metrics.items()
                    for stat in ("mean", "std", "train_mean")
                },
            }
        )
    pd.DataFrame(cv_rows).to_csv(
        reports_dir / "metricas_validacion_cruzada.csv",
        index=False,
        encoding="utf-8-sig",
    )
    artifacts.cv_fold_scores.to_csv(
        reports_dir / "metricas_cv_por_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grid_columns = [
        "params",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "rank_test_score",
    ]
    artifacts.logistic_grid_results[grid_columns].to_csv(
        reports_dir / "resultados_grid_logistica.csv",
        index=False,
        encoding="utf-8-sig",
    )
    artifacts.svm_grid_results[grid_columns].to_csv(
        reports_dir / "resultados_grid_svm.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_rows = [
        {"modelo": model_name, **metrics}
        for model_name, metrics in artifacts.results["test_metrics"].items()
    ]
    pd.DataFrame(test_rows).to_csv(
        reports_dir / "metricas_test.csv", index=False, encoding="utf-8-sig"
    )

    metadata = build_metadata(
        artifacts.modeling_table[MODEL_FEATURES],
        artifacts.results,
    )
    _write_json(models_dir / "metadata_modelo.json", metadata)
    create_plots(artifacts, figures_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y valida el proyecto final.")
    parser.add_argument("--input", type=Path, required=True, help="CSV original")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Carpeta raiz de la solucion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    artifacts = train_and_evaluate(raw)
    save_artifacts(artifacts, args.output_root)
    selected = artifacts.results["model_selection"]["selected_model"]
    f1 = artifacts.results["final_test_metrics"]["f1"]
    print(f"Modelo final: {selected}")
    print(f"F1 en test: {f1:.4f}")


if __name__ == "__main__":
    main()
