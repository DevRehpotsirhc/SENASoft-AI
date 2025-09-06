from transformers import pipeline


ai = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)


def clasificar(comentario: str) -> str:
    etiqueta = ["positivo", "negativo", "otro"]

    clasificacion = ai(comentario, candidate_labels=etiqueta)

    return "Positivo" if clasificacion["labels"][0] == "positivo" else "Negativo" if clasificacion["labels"][0] == "negativo" else "Parece que su comentario no puede clasificarse entre positivo o negativo" # type:ignore


comentario = "No me gusta Laura, pero en realidad sí"

print(clasificar(comentario))
