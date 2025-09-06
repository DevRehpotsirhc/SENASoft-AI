from transformers import pipeline
from datetime import datetime
import locale


locale.setlocale(locale.LC_TIME, 'es_CO.utf8')


ai = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")


def dia_actual(pregunta: str) -> str:
    etiquetas = ["pregunta sobre qué día es hoy", "otra"]

    resultado = ai(pregunta, candidate_labels=etiquetas)

    if resultado["labels"][0] == "pregunta sobre qué día es hoy" and resultado["scores"][0] > 0.8: # type:ignore
        hoy = datetime.now().strftime("%A %d de %B de %Y")
        return f"Hoy es {hoy}"
    else:
        return "Solo puedo responder qué día es hoy"


print(dia_actual("hola quiero saber qué día es hoy"))