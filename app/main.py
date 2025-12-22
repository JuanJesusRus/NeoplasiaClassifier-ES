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
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_form")
async def predict_form(request: Request, texto: str = Form("")):
    if not texto.strip():
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "El texto está vacío."}
        )

    resultado = inference_engine.predecir(texto)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "resultado": resultado,
            "input_text": texto
        }
    )


@app.post("/predict_file_form")
async def predict_file_form(request: Request, archivo: UploadFile = File(...)):
    contenido = (await archivo.read()).decode("utf-8")

    resultado = inference_engine.predecir(contenido)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "resultado": resultado,
            "input_text": contenido
        }
    )
