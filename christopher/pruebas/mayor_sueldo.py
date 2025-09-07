import pandas as pd
from transformers import pipeline

# 1. Cargamos el CSV
df = pd.read_csv("christopher/data/personas.csv")

# 2. LLM para NLP
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# 3. Función para responder
def responder(pregunta: str):
    if "mayor sueldo" in pregunta.lower():
        max_sueldo = df["sueldo"].max()
        personas = df[df["sueldo"] == max_sueldo]
        resultado = ", ".join(
            f"{row['nombre']} {row['apellido']} con sueldo {row['sueldo']}"
            for _, row in personas.iterrows()
        )
    elif "menor sueldo" in pregunta.lower():
        min_sueldo = df["sueldo"].min()
        personas = df[df["sueldo"] == min_sueldo]
        resultado = ", ".join(
            f"{row['nombre']} {row['apellido']} con sueldo {row['sueldo']}"
            for _, row in personas.iterrows()
        )
    else:
        resultado = "No entendí la pregunta en este prototipo."

    # --- Paso 2: LLM para embellecer la respuesta ---
    prompt = f"La respuesta es {resultado}"
    respuesta = generator(prompt, max_length=100)[0]["generated_text"]

    return respuesta

# Ejemplo
print(responder("¿Quién tiene el mayor sueldo?"))
