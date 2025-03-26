import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cargar modelo y tokenizer entrenados
modelo_path = "C:/Users/jesus/OneDrive - Universidad de Málaga/Cuarto/TFG/NeoplasiaClassifier-ES/output/roberta"
tokenizer = AutoTokenizer.from_pretrained(modelo_path)
model = AutoModelForSequenceClassification.from_pretrained(modelo_path)
model.eval()

# Usar CPU
device = torch.device("cpu")
model.to(device)

# Frase de ejemplo
texto ="Paciente mujer de 59 años diagnosticada en mayo de 2024 con carcinoma ductal infiltrante de mama izquierda, estadio IIA, receptores hormonales positivos y HER2 negativo. Se realizó tumorectomía con márgenes libres y ganglio centinela negativo. Actualmente en tratamiento adyuvante con hormonoterapia (tamoxifeno) y seguimiento por oncología médica. No presenta signos de recidiva ni metástasis en las pruebas de imagen. Buen estado general, ECOG 0. Sin antecedentes personales de otras neoplasias ni antecedentes familiares oncológicos relevantes. No fumadora, no consume alcohol. Control de tensión arterial y perfil lipídico dentro de rango. Sin efectos adversos destacables durante el tratamiento hasta la fecha."


# Tokenización
inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Predicción
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

# Mostrar resultado
clases = {0: "Una neoplasia", 1: "Múltiples neoplasias"}
print(f"🔍 Texto: {texto}")
print(f"📌 Predicción: {clases[pred]}")
print(f"Probabilidades: {probs.squeeze().tolist()}")
