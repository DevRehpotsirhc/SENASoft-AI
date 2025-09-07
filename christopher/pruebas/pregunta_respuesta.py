from transformers import pipeline
import locale


ai = pipeline(
    "question-answering", 
    model="deepset/xlm-roberta-base-squad2"
)

def responder(pregunta: list[str], contexto: list[str]) -> list:
    respuestas = []

    for p in pregunta:
        mejor = None
        score_max = 0

        for c in contexto:
            r = ai(question=p, context=c)

            if r["score"] > score_max:      # type:ignore
                mejor = r
                score_max = r["score"]      # type:ignore

        respuestas.append(mejor["answer"])     # type:ignore
    return respuestas

contexto = [
    "La capital de Colombia es Bogotá",
    "La capital de Venezuela es Caracas",
    "La capital de México es Ciudad de México",
    "La capital de Ecuador es Quito",
    "La capital de Estados Unidos es Washington D.C.",
]

preguntas = [
    "capital de Estados Unidos",
]

print(responder(preguntas, contexto))