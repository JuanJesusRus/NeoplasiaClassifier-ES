import torch
from lime.lime_text import LimeTextExplainer


NUM_SAMPLES = 100
BATCH_SIZE = 4


def explicar_texto(texto: str, model, tokenizer, max_length: int) -> list:
    class_names = ["Una neoplasia", "Múltiples neoplasias"]
    explainer = LimeTextExplainer(class_names=class_names)
    model.to("cpu")
    model.eval()

    def predict_proba(texts):
        texts = list(texts)
        batches = []

        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            batches.append(probs.detach().cpu())

        return torch.cat(batches, dim=0).numpy()

    pred_idx = int(predict_proba([texto])[0].argmax())
    exp = explainer.explain_instance(
        texto,
        predict_proba,
        num_features=10,
        num_samples=NUM_SAMPLES,
        labels=[pred_idx],
    )

    return [(feature, float(weight)) for feature, weight in exp.as_list(label=pred_idx)]
