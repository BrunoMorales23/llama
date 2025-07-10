import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from langchain_ollama import OllamaEmbeddings
#from langchain_community.vectorstores import Chroma
#from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json

#aca va la lectura del documento
#archivo = "C:/Users/bmorales/Downloads/BD-Adm&Finanzas-DS_Backup (1).xlsx"
archivo ="C:/Users/MarsuDIOS666/Desktop/llama/inputs/CSV UTF.csv"
#archivo = "C:/Users/MarsuDIOS666/Desktop/llama/inputs/BD-Adm&Finanzas-DS_Backup.csv"
#df = pd.read_excel(archivo)
df = pd.read_csv(archivo, sep=';', encoding='utf-8', index_col=False)
#df = pd.read_csv(archivo)
print(df)

#embeddings = OllamaEmbeddings(model="mxbai_embed_large")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="Testing",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        content = f"""
        Año - mes: {row.get('Año - mes', '')}
        Fecha: {row.get('Fecha', '')}
        Tipo: {row.get('Tipo', '')}
        Comprobante: {row.get('Comprobante', '')}
        Cliente: {row.get('Cliente', '')}
        """

        doc = Document(
            page_content=content.strip(),
            metadata={"id": str(i)},
        )

        documents.append(doc)
        ids.append(str(i))

    # Batching
    MAX_BATCH_SIZE = 5461
    total = len(documents)
    print(f"Agregando {total} documentos en batches de {MAX_BATCH_SIZE}...")

    for i in range(0, total, MAX_BATCH_SIZE):
        batch_docs = documents[i:i + MAX_BATCH_SIZE]
        batch_ids = ids[i:i + MAX_BATCH_SIZE]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"Batch agregado: {i} al {min(i + MAX_BATCH_SIZE, total)}")

    print(f"{total} documentos agregados a la base.")

#if add_documents:
#    vector_store.add_documents(documents=documents, ids=ids)
#    print(f"{len(documents)} documentos agregados a la base.")

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": df.shape[0],
        "score_threshold": 0.6
    }
)

model = OllamaLLM(model = "llama3.2", temperature=0.1)
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
- El formato preferido es **texto limpio y ordenado línea por línea**.
- **No expliques ni interpretes los datos**.
- En caso de múltiples coincidencias, **devuelve todas las coincidencias encontradas** que sea claro y preciso.
- **No inventes respuestas ni completes información faltante**.

Tu respuesta debe limitarse únicamente a lo solicitado. Todo lo que exceda eso debe ser omitido.
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

all_docs = vector_store.get()
content = all_docs['documents']

while True:
    print("\n\n--------------------------------")
    quest = input("Pregunta lo que quieras. (Presiona X para salir)... ")
    print("\n\n")
    if quest == "X":
        break

    result = chain.invoke({"content": content, "question": quest})
    print(result)