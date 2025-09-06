import pandas as pd
from transformers import pipeline
import re

# Cargar CSV
df = pd.read_csv("usuarios_usuario.csv")

# Definir funciones de consulta
def contar_rango(inicio: int, fin: int) -> int:
    return len(df.iloc[inicio-1:fin])

def nombre_num(num: int):
    return df.iloc[num-1]["usuario"]

def nombre_por_id(id_usuario: int):
    fila = df.loc[df["id_usuario"] == id_usuario]
    return fila.iloc[0]["usuario"] if not fila.empty else None

# Modelo Hugging Face
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def responder(pregunta: str) -> str:
    prompt = f"""
Tienes un dataframe con columnas: {list(df.columns)}.
Usa exclusivamente estas funciones para responder:
- contar_rango(inicio:int, fin:int)
- nombre_num(num:int)
- nombre_por_id(id_usuario:int)

Responde SOLO con una instrucción en Python válida.
NO des explicaciones, NO repitas la pregunta.

Ejemplos:
Pregunta: ¿Cuántos usuarios hay desde el primero al penúltimo?
Respuesta: contar_rango(1, len(df)-1)

Pregunta: ¿Cuántos usuarios hay del quinto al último?
Respuesta: contar_rango(5, len(df))

Pregunta: ¿Cuál es el nombre del cuarto usuario?
Respuesta: nombre_num(4)

Pregunta: ¿Quién es el usuario con id 41?
Respuesta: nombre_por_id(41)

Ahora responde:
Pregunta: "{pregunta}"
Respuesta:
"""
    salida = generator(prompt, max_new_tokens=64)[0]["generated_text"].strip()

    # Extraer función con regex
    match = re.search(r"(contar_rango\(.*?\)|nombre_num\(.*?\)|nombre_por_id\(.*?\))", salida)
    if not match:
        return f"No entendí la salida: {salida}"

    instruccion = match.group(0)

    # Correcciones conocidas
    correcciones = {
        "name_num": "nombre_num",
        "user_id": "nombre_por_id",
        "id_usuario": "nombre_por_id",
    }

    for mal, bien in correcciones.items():
        instruccion = instruccion.replace(mal, bien)
        
    # Arreglar paréntesis si falta
    if instruccion.count("(") > instruccion.count(")"):
        instruccion += ")"

    print(f"Código generado: {instruccion}")

    try:
        return eval(instruccion)
    except Exception as e:
        return f"Error al ejecutar {instruccion}: {e}"



# --- Ejemplos ---
print(responder("¿Cuántos usuarios hay?"))
print(responder("Dime el nombre del cuarto usuario"))
print(responder("usuario con id 41"))