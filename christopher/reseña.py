from transformers import pipeline


ai = pipeline(
    "sentiment-analysis", # type:ignore
    model="nlptown/bert-base-multilingual-uncased-sentiment"
) # type:ignore


textos = [
    "La película fue increíble, me encantó",
    "El servicio del restaurante fue terrible",
    "El celular tiene muy buenas prestaciones, pero la cámara podría ser mejor"
]


resultados = ai(textos)


for texto, resultado in zip(textos, resultados):
    if "3" in resultado["label"]:
        r = "Regular"
    else:
        r = "Positivo" if ("4" in resultado["label"] or "5" in resultado["label"]) else "Negativo"

    print(f"\nComentario: {texto}\nReseña: {resultado}")