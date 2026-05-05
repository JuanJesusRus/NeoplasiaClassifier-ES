# Clasificador de Neoplasias mediante NLP

Aplicación web desarrollada con FastAPI para la clasificación de historias clínicas en:

- **Una neoplasia**
- **Múltiples neoplasias**

Incluye un sistema de **explicabilidad basado en LIME** para interpretar las predicciones del modelo.

---

##  Descripción

El sistema utiliza modelos basados en Transformers entrenados sobre texto clínico para:

- Clasificación binaria (una vs múltiples neoplasias)
- Clasificación multietiqueta mediante un enfoque en cascada
- Explicación de predicciones mediante LIME

La aplicación permite introducir texto clínico o subir archivos para obtener predicciones.

---

##  Estructura del proyecto

NeoplasiaClassifier-ES/
├── app/
│ ├── main.py
│ ├── inference.py
│ ├── lime_utils.py
│ ├── templates/
│ └── static/
├── models/ # Carpeta donde deben colocarse los modelos (no incluidos)
├── scripts/ # Scripts auxiliares de experimentación (no necesarios para la app)
├── config_app.yaml
├── requirements.txt
└── README.md


---

##  Requisitos

Instalar dependencias:
pip install -r requirements.txt


---

##  Ejecución

Ejecutar la aplicación:
uvicorn app.main:app --reload

Abrir en el navegador:
http://127.0.0.1:8000


---

##  Modelos

 **Los modelos no se incluyen en el repositorio debido a su tamaño (~500MB / 1Gb cada uno).**

Deben colocarse manualmente en la carpeta `models/` con la siguiente estructura:
models/
├── binario/
├── mama/
└── resto/


Las rutas de los modelos se configuran en el archivo `config_app.yaml`.

---

##  Explicabilidad (LIME)

La aplicación permite generar explicaciones de las predicciones mediante LIME:

- Identifica las palabras que más influyen en la predicción
- Permite interpretar el comportamiento del modelo

 La generación de explicaciones es computacionalmente costosa y puede tardar varios segundos.

Por este motivo, su uso es opcional en la interfaz.

---

##  Limitaciones

- Sensibilidad a estructuras de negación en el texto clínico
- Posibles sesgos derivados del dataset de entrenamiento
- Tiempo de respuesta elevado al utilizar LIME
- Tamaño elevado de los modelos

---

##  Aplicación potencial

El sistema se plantea como un prototipo de apoyo a la decisión clínica.

Para su integración en entornos hospitalarios reales sería necesario:

- Validación clínica del modelo
- Integración con sistemas de historia clínica electrónica
- Cumplimiento de normativas de protección de datos

---

##  Notas

- La aplicación está diseñada para ejecutarse en entorno local
- Los modelos se cargan en memoria al iniciar la aplicación
- La carpeta `scripts/` contiene utilidades de desarrollo y no es necesaria para el uso de la aplicación