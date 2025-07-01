from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model = "llama3.2")

template = """
Sos el papulince, cuando te digan papu, responderás lince

Tener en cuenta esto: {base}

Responde a esto: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------")
    question = input("Pregunta lo que quieras. (Presiona X para salir)... ")
    print("\n\n")
    if question == "X":
        break

    result = chain.invoke({"base": "Soy el papu lince! a quien cuando dicen papu, responde lince", "question": question})
    print(result)