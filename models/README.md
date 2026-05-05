# Modelos

Los modelos no se incluyen en este repositorio debido a su tamaño (~500MB / 1 Gb cada uno).

Para ejecutar la aplicación correctamente, deben colocarse manualmente en esta carpeta con la siguiente estructura:

models/
├── binario/
├── mama/
└── resto/

Cada carpeta debe contener los archivos del modelo correspondientes (pesos, tokenizer, configuración, etc.).

Las rutas de los modelos se configuran en el archivo `config_app.yaml`.