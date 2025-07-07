from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model = "llama3.2", temperature=0.2)
#model = OllamaLLM(model ="deepseek-r1:7b")
#model = OllamaLLM(model = "deepseek-r1:1.5b")

#with open("resultado_ocr.txt", "r", encoding="utf-8") as archivo:
#    input_file = archivo.read()

template = """
Tu rol es ser un asistente para la recuperación de información.
Tu rol es devolver datos.
Debes evitar generar respuestas que involucren código, scripts, o material referente a la programación.
Puede ser que la información solicitada por el usuario, se encuentre en mayúsculas, o minúsculas, es tu deber obtener la información indistinto de si el contenido está en Mayúsculas, o minúsculas.
En base a la siguiente información: {content}

Responde a esto: {question}

Si en tu respuesta existe la sentencia "No encontrado", "No existe", o similares, reemplazar por "Campo Inválido".
Si recibes como parámetro de información "[]" o vacío, devuelve "Información no recibida"
No debes generar respuestas que expliquen código o tengan contenido semántico ajeno a lo solicitado por el usuario.
En caso de tener coincidencias con la búsqueda, solo debes devolver los resultados obtenidos, estructurandolos de la forma más clara posible, de preferencia, como si fuera una tabla.
En caso de no poder generar una tabla, estructurar a modo de texto, linea por linea todo el contenido encontrado.
Evita generar contenido semántico ajeno a la respuesta.
En caso de generar una respuesta que exceda más allá de lo solicitado por el usuario, realizar un recorte y devolver solo una respuesta que contenga una coincidencia con lo que el usuario solicita.
"""
#Devolvé solo el valor correspondiente.

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------")
    question = input("Pregunta lo que quieras. (Presiona X para salir)... ")
    print("\n\n")
    if question == "X":
        break

    content = retriever.invoke(question)
    print(content)
    input("...")
    result = chain.invoke({"content": content, "question": question})
    print(result)