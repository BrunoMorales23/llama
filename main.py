import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import json

model = OllamaLLM(model = "llama3.2", temperature=0.2)
#model = OllamaLLM(model ="deepseek-r1:7b")
#model = OllamaLLM(model = "deepseek-r1:1.5b")

#with open("resultado_ocr.txt", "r", encoding="utf-8") as archivo:
#    input_file = archivo.read()

template =""" Tu único rol es ser un asistente de recuperación de información.

Debes limitarte exclusivamente a devolver datos directamente relacionados con la consulta del usuario, **sin agregar explicaciones, sin generar código, sin scripts, sin suposiciones, ni contenido adicional**. No estás autorizado a generar ningún bloque de código ni a razonar como programador.

El contenido que recibís como información puede estar en mayúsculas, minúsculas o mezclado. Debes interpretar correctamente sin importar el formato.

Debes responder únicamente en base a la siguiente información recibida: {content}

Y a la siguiente pregunta del usuario: {question}

📌 Instrucciones obligatorias:
- Si el contenido recibido es vacío o igual a "[]", responde: **"Información no recibida"**.
- Si no hay coincidencias relevantes en los datos, responde: **"Campo Inválido"**.
- Si encontrás coincidencias, devuélvelas **sin ningún texto adicional**.
- El formato preferido es **en forma de tabla**, pero si no es posible, usá texto limpio y ordenado línea por línea.
- **No expliques ni interpretes los datos**.
- En caso de múltiples coincidencias, **devolvé solo una** que sea clara y representativa.
- **No inventes respuestas ni completes información faltante**.

Tu respuesta debe limitarse únicamente a lo solicitado. Todo lo que exceda eso debe ser omitido.
"""
#Devolvé solo el valor correspondiente.

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------")
    quest = input("Pregunta lo que quieras. (Presiona X para salir)... ")
    print("\n\n")
    if quest == "X":
        break

    
    content = retriever.invoke(quest)
    print(content)
    input("...")
    result = chain.invoke({"content": content, "question": quest})
    print(result)
    #respuesta_json = model.invoke(prompt.format_messages(question=quest))