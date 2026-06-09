# Modelos

Los modelos entrenados no se incluyen en el repositorio debido a su tamano. La carpeta `models/` se entrega solo con este archivo de instrucciones.

## Descarga

Descargue los modelos desde el siguiente enlace de Drive:

```txt
https://drive.google.com/file/d/1wLKNNolqbJa934kWXcT2a4e1VP2GGELT/view?usp=sharing
```

## Estructura esperada

Una vez descargados y descomprimidos, coloque los modelos dentro de esta carpeta respetando la siguiente estructura:

```txt
models/
|-- binario/
|   `-- galen_1/
|-- mama/
|   |-- best_model/
|   `-- label2idx.json
`-- resto/
    |-- best_model/
    `-- label2idx.json
```

Las rutas ya estan configuradas en `app/config_app.yaml` mediante rutas relativas desde la carpeta `app/`.
