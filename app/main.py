from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from inference import NeoplasiaInference
import yaml
from fastapi.staticfiles import StaticFiles


with open("config_app.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

inference_engine = NeoplasiaInference(config)

app = FastAPI(title="Clasificador de Neoplasias")

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "input_explicacion": False}
    )


@app.post("/predict_form")
async def predict_form(
    request: Request,
    texto: str = Form(""),
    mode: str = Form("binario"),
    explicacion: str | None = Form(None),
):
    generar_explicacion = explicacion == "on"

    if not texto.strip():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "El texto está vacío.",
                "input_text": texto,
                "input_mode": mode,
                "input_explicacion": generar_explicacion,
            }
        )

    resultado = inference_engine.predecir(
        texto,
        mode=mode,
        generar_explicacion=generar_explicacion,
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "resultado": resultado,
            "input_text": texto,
            "input_mode": mode,
            "input_explicacion": generar_explicacion,
        }
    )


@app.post("/predict_file_form")
async def predict_file_form(
    request: Request,
    archivo: UploadFile = File(...),
    mode: str = Form("binario"),
    explicacion: str | None = Form(None),
):
    generar_explicacion = explicacion == "on"
    contenido = (await archivo.read()).decode("utf-8")

    resultado = inference_engine.predecir(
        contenido,
        mode=mode,
        generar_explicacion=generar_explicacion,
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "resultado": resultado,
            "input_text": contenido,
            "input_mode": mode,
            "input_explicacion": generar_explicacion,
        }
    )
