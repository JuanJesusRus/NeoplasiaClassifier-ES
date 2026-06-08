# Clasificador de Neoplasias mediante Procesamiento de Lenguaje Natural

Aplicación web desarrollada con FastAPI para la clasificación automática de historias clínicas electrónicas.

Funcionalidades principales:

- Clasificación binaria (una neoplasia frente a múltiples neoplasias).
- Clasificación multietiqueta mediante un enfoque en cascada.
- Explicabilidad de predicciones mediante LIME.

## Estructura del proyecto

NeoplasiaClassifier-ES/
├── app/
├── models/
├── script/
└── README.md

## Instalación rápida

1. Crear entorno virtual.
2. Instalar dependencias:

pip install -r app/requirements.txt

3. Copiar los modelos en la carpeta `models/`.

## Ejecución

cd app

python -m uvicorn main:app --reload

## Modelos

Los modelos entrenados no se incluyen en el repositorio debido a su tamaño.

La estructura esperada es:

models/
├── binario/
├── mama/
└── resto/

Las rutas ya se encuentran configuradas mediante rutas relativas en `app/config_app.yaml`.

## Documentación

La guía completa de instalación, configuración y uso se encuentra en:

- Manual de Instalación y Uso.pdf

## Autor

Juan Jesús Rus Muñoz

Trabajo Fin de Grado – Universidad de Málaga