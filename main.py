from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model = "llama3.2")

with open("resultado_ocr.txt", "r", encoding="utf-8") as archivo:
    input_file = archivo.read()

base = """
Tu tarea es identificar un valor específico dentro de un texto proporcionado por el usuario. El usuario te indicará cuál de los siguientes campos desea obtener:

- "vencimiento"
- "total a pagar"
- "LSP"
- "Cuenta de Servicios"
- "Servicios"
- "Categoría del Usuario"

Debés buscar únicamente el valor asociado al campo solicitado, sin inferencias ni explicaciones adicionales. Si no encontrás el valor, respondé únicamente con: "No encontrado".
Si lo solicitado por el usuario no cumple el criterio de búsqueda. Devolver: "Campo no válido" y anular el resto de reglas.

"Servicios", puede ser encontrado entre "Categoria de Usuario" y "Período de Facturación" y su valor suele ser "AGUA", "AGUA POTABLE", "CLOACA" y la iteración de unas y otras, por ejemplo: "AGUA Y CLOACAS"
"Total a Pagar", es siempre un número. Completar obligatoriamente con números. En caso de que exista "o" o "O" en la respuesta, debe ser reemplazado por "0".

A continuación, se te indicará el campo deseado y el texto de entrada. Debes identificar la palabra clave usada por el usuario, la cuál será una de las de la lista mencionada previamente.
Devolvé solo el valor correspondiente.
"""

template = """
Tu rol es el siguiente: {base}
La información que debes analizar es: {input_file}
Responde a esto: {question}

Vuelve a validar si lo solicitado cumple con el criterio de búsqueda:
- "vencimiento"
- "total a pagar"
- "LSP"
- "Cuenta de Servicios"
- "Servicios"
- "Categoría del Usuario"

Debés buscar únicamente el valor asociado al campo solicitado, sin inferencias ni explicaciones adicionales. Si no encontrás el valor, respondé únicamente con: "No encontrado".
Si lo solicitado por el usuario no cumple el criterio de búsqueda. Devolver: "Campo no válido" y anular el resto de reglas.

En caso de no tener una respuesta concreta, abstenerse de generar contenido y responder solo con la siguiente literal: "Búsqueda no realizada"
En caso de que la información obtenida tenga caracteres fuera de lugar, debes corregirlo.
Ejemplos de preguntas: "¿Cuál es el total a pagar?","¿Cuándo es el vencimiento?","Dame el valor de LSP","Dame el número de LSP","Quiero saber la Cuenta de Servicios","¿A qué categoría corresponde el usuario?
Ejemplos de respuestas: "Total a pagar: $000.000,00", "Vencimiento: 00/00/0000", "LSP: 0000X00000000", "Cuenta de Servicios: 0000000", "Categoria de Usuario: RESIDENCIAL"
Responder concretamente, sin generar oraciones no solictadas.

Si en tu respuesta no existe la sentencia "No encontrado", "No existe", o similares, reemplazar por "Campo Inválido".
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------")
    question = input("Pregunta lo que quieras. (Presiona X para salir)... ")
    print("\n\n")
    if question == "X":
        break

    result = chain.invoke({"base": {base}, "question": question, "input_file": input_file})
    print(result)