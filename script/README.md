# Scripts auxiliares

Esta carpeta contiene scripts utilizados durante el desarrollo, entrenamiento, evaluación y análisis de los modelos. No son necesarios para instalar ni ejecutar la aplicación web de inferencia.

Importante: no todos los archivos de esta carpeta forman parte del flujo final. Algunos se conservan como material histórico del desarrollo, pueden estar obsoletos o no usarse actualmente. Los scripts relevantes para entender o reproducir el trabajo son los indicados en la sección de scripts principales.

Para usar la aplicación, consulte el manual principal y ejecute la app desde la carpeta `app/`.

## Scripts principales

Estos scripts son los más reutilizables porque tienen una estructura más general o permiten configuración mediante argumentos o archivos YAML:

- `neoplasia.py`: script unificado para entrenamiento y evaluación del clasificador binario.
- `filtrar2.py`: preparación de los conjuntos finales usados para entrenamiento, validación y test.
- `train_multilabel.py`: entrenamiento del clasificador multietiqueta.
- `test_multilabel.py`: evaluación del clasificador multietiqueta.
- `predict_cascade_tune.py`: predicción y ajuste de umbrales para el enfoque en cascada.
- `creartablaTuningumbrales.py`: generación de tablas resumen a partir de métricas guardadas.


## Scripts experimentales o históricos

Estos scripts se conservaron como apoyo al desarrollo del TFG, pero pueden contener rutas absolutas, depender de archivos locales, estar obsoletos o corresponder a versiones anteriores del flujo experimental:

- `division_train.py`: versión previa de división de datos; no corresponde al flujo final.
- `eda_neoplasias.py`: análisis exploratorio del conjunto de datos.
- `contadorNeoplasiasTotales.py`: conteo y resumen de combinaciones de neoplasias.
- `filtrarLongitud.py`: utilidad de filtrado por longitud de texto.
- `entrenamiento_cv.py`: entrenamiento binario con validación cruzada.
- `optuna_hiperparametros.py`: pruebas de ajuste de hiperparámetros con Optuna.
- `model.py`: versión previa de entrenamiento binario.
- `predict_neoplasias.py`: evaluación puntual de modelos binarios.
- `predict_neoplasias_coleccion.py`: evaluación usando colecciones de etiquetas.
- `predict_cascade.py`: versión previa de predicción en cascada.
- `testMetricasCombinacion.py`: cálculo de métricas por combinación de neoplasias.
- `graficarPRcombinacion.py`: generación de gráficas a partir de métricas por combinación.
- `explicabilidad.py`: pruebas de explicabilidad con LIME fuera de la aplicación web.

## Nota

Antes de reutilizar cualquier script, revise sus rutas de entrada, salida, datos y modelos. La aplicación web no depende de esta carpeta para funcionar.