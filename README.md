# Proyecto final — Clasificación de presentaciones CUM como muestra médica

Aplicación académica de aprendizaje automático que prioriza presentaciones del **Código Único de Medicamentos (CUM)** cuya estructura es compatible con la condición de *muestra médica*. La predicción sirve como señal para orientar una revisión documental; **no reemplaza la clasificación oficial ni una decisión regulatoria de INVIMA**.

**Integrantes:** Brayam Arboleda Diaz y Saul Dario Gomez  
**Curso:** Aprendizaje de Máquinas  
**Fuente:** [Código Único de Medicamentos Vigentes — Datos Abiertos Colombia](https://www.datos.gov.co/Salud-y-Protecci-n-Social/C-DIGO-NICO-DE-MEDICAMENTOS-VIGENTES/i7cb-raxc/about_data)  
**Publicador:** INVIMA  
**Corte reproducido:** 21 de septiembre de 2026  
**Repositorio:** [Badiaz9226/Proyecto-Final-ML](https://github.com/Badiaz9226/Proyecto-Final-ML)  
**Google Colab público:** [Abrir el notebook ejecutado en Colab](https://colab.research.google.com/github/Badiaz9226/Proyecto-Final-ML/blob/main/notebooks/Proyecto_Final_ML_CUM.ipynb)

## Problema y uso previsto

El proyecto responde la pregunta: **¿una nueva presentación CUM tiene un patrón estructural semejante al de las presentaciones registradas como muestra médica?** La clase positiva es `muestramedica = Sí`. La salida permite priorizar casos para revisión de consistencia de empaque y documentación.

El alcance es deliberadamente limitado:

- Es una herramienta de apoyo académico, no un sistema clínico ni regulatorio.
- El modelo aprende asociaciones del corte estudiado; no demuestra causalidad.
- Un resultado negativo no certifica que una presentación no sea muestra médica.
- El puntaje mostrado en la aplicación **no está calibrado**. Sirve para ordenar o priorizar casos, pero no debe leerse como probabilidad, certeza ni nivel de riesgo.

## Comprensión y calidad de los datos

El archivo original contiene **155.807 filas y 29 columnas**, pero su grano no es una presentación: una misma combinación `(expedientecum, consecutivocum)` se repite por ingredientes, vías de administración, roles y establecimientos. Tras consolidar el grano correcto quedaron **65.335 presentaciones CUM** pertenecientes a **9.520 expedientes**. Las **90.472 filas adicionales** del origen son expansión relacional, no observaciones independientes.

La variable objetivo presenta **14.092 positivos (21,57 %) y 51.243 negativos (78,43 %)**. Por ello, *accuracy* aislada sería engañosa; la métrica principal es **F1**, acompañada por precisión, recall, accuracy balanceada, ROC-AUC y PR-AUC.

Riesgos encontrados y tratamiento:

| Riesgo | Evidencia | Decisión |
|---|---:|---|
| Duplicación lógica | 83,40 % de las filas pertenece a CUM repetidos | Consolidar a una fila por presentación |
| Fuga por partición aleatoria | 73,41 % de las filas de test compartiría CUM con train | Separar por `expedientecum` |
| Fuga entre presentaciones hermanas | 95,22 % de los CUM de test compartiría expediente con train en un split simple por CUM | Usar `StratifiedGroupKFold` con expediente como grupo |
| Fuga directa desde texto | Buscar “muestra médica” en producto/descripción logra 97,42 % de accuracy | Excluir `producto` y `descripcioncomercial` |
| Campos posteriores al resultado | Estado, inactivación y vencimiento revelan eventos del ciclo de vida | Excluir estado y todas las fechas |
| Valores ausentes | IUM 79,16 %, fecha de inactivación 75,92 %, vencimiento 70,71 % | No usar esos campos; imputar únicamente dentro de cada fold |
| Fechas centinela | 31.473 valores con año 3000, además de 1000, 1900 y 9999 | No transformar esos valores en señal predictiva |
| Clase minoritaria | 21,57 % de positivos | Sobremuestreo solo sobre train y dentro de cada fold |

El análisis completo está en [`reports/reporte_ydata_profiling.html`](reports/reporte_ydata_profiling.html), y los controles reproducibles están en `reports/calidad_datos.csv`, `reports/diccionario_datos_original.csv`, `reports/asociaciones_variables.csv` y `reports/resumen_calidad.json`.

## Variables del modelo

El contrato de entrada contiene exactamente **18 variables estructurales** que también valida la aplicación:

- Numéricas: `cantidad_cum`, `numero_filas_fuente`, `numero_cantidades_ingrediente_distintas`, `numero_principios_activos`, `numero_vias_administracion`, `numero_codigos_atc`, `numero_unidades_medida`, `numero_unidades_referencia`, `numero_tipos_rol` y `numero_actores`.
- Categóricas simples: `forma_farmaceutica`, `codigo_concentracion`, `unidad_cum` y `modalidad`.
- Categóricas multivalor: `vias_administracion`, `unidades_medida`, `familias_atc` y `tipos_rol`.

No se usan nombres comerciales, descripciones, identificadores, titular, fechas ni estado regulatorio. Las familias ATC se reducen a su primera letra y las unidades de medida se normalizan antes de agregarse.

## Metodología CRISP-DM

1. **Comprensión del negocio:** se definió una alerta de priorización revisable, con usuario, alcance y límites explícitos.
2. **Comprensión de los datos:** se auditó el diccionario, grano, duplicación, faltantes, codificación, valores centinela, distribución del objetivo y riesgo de sesgo.
3. **Preparación:** se consolidó una fila por CUM, se normalizaron categorías y se conservaron solo variables conocidas al clasificar una presentación. El test no participa en ninguna transformación aprendida.
4. **Modelado:** se compararon cinco modelos clásicos —regresión logística, SVM lineal, MLP, árbol y KNN— y tres ensambles —votación, bagging y AdaBoost—. También se ajustaron hiperparámetros de regresión logística y SVM con búsqueda en cuadrícula.
5. **Evaluación:** la selección se cerró con F1 en validación cruzada agrupada de cinco folds sobre el 70 % de entrenamiento. Solo después se abrió una vez el 30 % de test para el reporte final.
6. **Despliegue:** imputación, transformación `log1p`, escalado, codificación, reducción para KNN, balanceo y clasificador se integran en pipelines reproducibles. El artefacto final se serializa completo para que Streamlit ejecute exactamente el mismo flujo.

### Control de fuga y diseño de validación

La separación 70/30 se forma con diez folds estratificados y agrupados: tres conforman test y siete entrenamiento. El solapamiento de expedientes entre ambos conjuntos debe ser **cero**. Dentro del 70 % se realiza validación cruzada agrupada de cinco folds. Imputadores, escaladores, vocabularios, codificación, reducción dimensional y `RandomOverSampler` se ajustan solo con la porción de entrenamiento de cada fold.

La regresión logística prueba `C ∈ {0,1; 1; 10}` y regularización L2/L1. La SVM lineal prueba el mismo rango de `C` y pérdidas `hinge`/`squared_hinge`. La selección estadística usa folds alineados, Friedman y comparaciones pareadas con corrección de Holm. Se exige una mejora práctica superior a **0,02 F1** frente a la regresión logística ajustada; si no aparece evidencia suficiente, se prefiere el modelo más simple e interpretable. Con solo cinco folds, los valores *p* se consideran evidencia auxiliar, no una prueba definitiva.

## Resultados

<!-- RESULTADOS_MODELO_INICIO -->
El conjunto de entrenamiento contiene **45.740 presentaciones (6.706 expedientes)** y el test bloqueado **19.595 presentaciones (2.814 expedientes)**, con prevalencias positivas de 21,58 % y 21,55 %, respectivamente. El solapamiento de expedientes es **0**.

| Modelo | F1 CV agrupada, media ± DE | F1 en test |
|---|---:|---:|
| Regresión logística | 0,4846 ± 0,0097 | 0,4890 |
| SVM lineal | 0,4831 ± 0,0086 | 0,4866 |
| Red neuronal MLP | 0,4918 ± 0,0065 | 0,4924 |
| Árbol de decisión | 0,4880 ± 0,0114 | 0,4908 |
| K vecinos más cercanos | 0,4700 ± 0,0098 | 0,4589 |
| Votación | 0,5016 ± 0,0066 | 0,5059 |
| **Bagging** | **0,5061 ± 0,0032** | **0,5082** |
| Boosting AdaBoost | 0,4862 ± 0,0128 | 0,4844 |
| Regresión logística ajustada | 0,4851 ± 0,0104 | 0,4889 |
| SVM lineal ajustada | 0,4874 ± 0,0113 | 0,4858 |

Se seleccionó **Bagging** antes de abrir el test. Superó a la regresión logística ajustada por **0,0211 F1** en validación, por encima del margen práctico de 0,02. La prueba de Friedman indicó diferencias globales (`p = 0,000229`) y la comparación pareada Bagging–logística ajustada conservó significancia tras corrección de Holm (`p ajustado = 0,0303`). La mejor cuadrícula logística usó `C = 0,1` y L2; la SVM ajustada usó `C = 0,1` y pérdida `hinge`.

Desempeño final de Bagging en el test no visto:

| Accuracy | Precisión | Recall | F1 | Accuracy balanceada | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 0,7055 | 0,3970 | 0,7061 | **0,5082** | 0,7057 | 0,7799 | 0,4900 |

La matriz de confusión es `[[10.842, 4.530], [1.241, 2.982]]`: se detectaron 2.982 de los 4.223 positivos del test. La accuracy de la línea base mayoritaria es mayor (0,7845), pero su recall y F1 son cero porque nunca identifica una muestra médica; este contraste confirma por qué F1 y las métricas por clase son las referencias correctas.
<!-- RESULTADOS_MODELO_FIN -->

La tabla comparativa completa está en `reports/metricas_validacion_cruzada.csv` y `reports/metricas_test.csv`; los puntajes por fold están en `reports/metricas_cv_por_fold.csv`. La selección del modelo no se modifica después de consultar el test.

## Estructura del repositorio

```text
.
├── app.py                                  # aplicación Streamlit
├── notebooks/
│   └── Proyecto_Final_ML_CUM.ipynb         # desarrollo incremental con prompts
├── data/
│   ├── codigo_unico_medicamentos_vigentes_20260921.csv
│   └── dataset_modelado_cum.csv
├── models/
│   ├── pipeline_muestra_medica.joblib       # pipeline completo para despliegue
│   └── metadata_modelo.json                 # contrato y opciones de entrada
├── reports/                                 # perfil, métricas y evidencia auditable
├── figures/                                 # gráficos y captura del despliegue público
├── docs/
│   └── Formato_Entrega_Proyecto_Integrador_CUM.docx
├── src/
│   ├── data_quality.py                      # auditoría de datos
│   ├── train_model.py                       # preparación, entrenamiento y evaluación
│   ├── transformers.py                      # transformación categórica reutilizable
│   └── integration_check.py                 # prueba extremo a extremo
├── requirements.txt                         # dependencias de producción
├── requirements-dev.txt                     # dependencias analíticas
└── runtime.txt                               # versión de Python para la nube
```

## Reproducción local

Requisitos: Python 3.13 y el CSV oficial guardado como `data/codigo_unico_medicamentos_vigentes_20260921.csv`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Auditoría, entrenamiento y prueba de integración:

```bash
python -m src.data_quality
python -m src.train_model \
  --input data/codigo_unico_medicamentos_vigentes_20260921.csv \
  --output-root .
python -m src.integration_check
```

La última orden debe finalizar con todos los controles en `OK` y crear `reports/prueba_integracion.csv`.

Ejecución de la aplicación:

```bash
streamlit run app.py
```

Abra la dirección local que muestre Streamlit, complete las 18 variables y pulse **Clasificar presentación**.

## Despliegue en Streamlit Community Cloud

Aplicación pública verificada: **https://proyecto-final-ml.streamlit.app**. La evidencia visual de una predicción funcional está en `figures/captura_streamlit.png` y también quedó insertada en el formato de entrega.

1. Publique este repositorio en GitHub con `app.py`, `models/`, `src/`, `requirements.txt` y `runtime.txt`.
2. En Streamlit Community Cloud seleccione **Create app**.
3. Elija `Badiaz9226/Proyecto-Final-ML`, rama `main`, archivo principal `app.py`.
4. Despliegue sin secretos: la aplicación solo usa artefactos versionados y rutas relativas a su propia carpeta.
5. Compruebe una predicción y las pestañas de metodología, desempeño y diccionario.

No se requieren rutas locales, Google Drive montado ni variables privadas. El archivo `requirements.txt` fija las versiones con las que se serializó el pipeline para evitar incompatibilidades entre `joblib` y scikit-learn.

## Limitaciones y uso responsable

- La fuente es un corte observacional y puede contener errores de captura, cambios de catálogo y categorías poco frecuentes.
- El sobremuestreo mejora el aprendizaje de la clase minoritaria, pero cambia la prevalencia vista durante el ajuste; por eso el puntaje del modelo **no está calibrado**.
- El ajuste de hiperparámetros y su resumen reutilizan los mismos cinco folds; no equivalen a validación cruzada anidada.
- La calidad fuera del periodo y de la población observada no fue demostrada. Ante actualizaciones de INVIMA se debe repetir auditoría, entrenamiento y validación.
- La aplicación no procesa datos personales y no debe usarse para decisiones clínicas, comerciales automáticas o sancionatorias.

## Trazabilidad

- Dataset oficial: recurso Socrata `i7cb-raxc`, Datos Abiertos Colombia.
- SHA-256 del snapshot: `43cd984ded90ddb8b0bd3d6abdb60a0f7628cd8975cab08fdfb6b61c09e18e6a`.
- Semilla global: `42`.
- Grano analítico: una fila por `(expedientecum, consecutivocum)`.
- Grupo de partición: `expedientecum`.
- Métrica de selección: F1 de la clase `Sí`.
- Evidencia del perfil: `reports/reporte_ydata_profiling.html`.
- Evidencia de integración: `reports/prueba_integracion.csv`.
- El cuaderno contiene los prompts usados paso a paso y no incluye resultados inventados: al ejecutarlo en Colab reproduce los artefactos desde la fuente.
