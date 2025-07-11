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
from tqdm import tqdm

#aca va la lectura del documento
#archivo = "C:/Users/bmorales/Downloads/BD-Adm&Finanzas-DS_Backup (1).xlsx"
archivo = "C:/Users/bmorales/OneDrive - rmrconsultores.com/Escritorio/llama/SHORTVER.csv"
#archivo = "C:/Users/MarsuDIOS666/Desktop/llama/inputs/BD-Adm&Finanzas-DS_Backup.csv"
#df = pd.read_excel(archivo)
df = pd.read_csv(archivo, sep=None, encoding="utf-8", index_col=False, engine="python")
df.columns = df.columns.str.strip()
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

# BATCH_SIZE = 100

# # Sinónimos para cada campo
SYNONYMS = {
    "Año - mes": ["período", "año‑mes", "mes contable"],
    "Cód. Cliente": ["código de cliente", "identificador de cliente"],
    "Razón Social": ["razón social", "nombre del cliente", "empresa"],
    "Fecha Emisión Créd.": ["fecha emisión crédito", "fecha crédito"],
    "Tipo Comprob. Créd.": ["tipo comprobante crédito", "tipo doc. crédito"],
    "Comprobante Créd.": ["comprobante crédito", "número doc. crédito"],
    "Tipo Comprob. Déb.": ["tipo comprobante débito", "tipo doc. débito"],
    "Comprobante Déb.": ["comprobante débito", "número doc. débito"],
    "Fecha Emisión Déb.": ["fecha emisión débito", "fecha débito"],
    "Fecha Vto. Créd.": ["vencimiento crédito", "fecha vto. crédito"],
    "Fecha Vto. Déb.": ["vencimiento débito", "fecha vto. débito"],
    "Importe Créd.": ["importe crédito", "monto crédito"],
    "Importe Aplicado": ["importe aplicado", "monto aplicado"],
    "Saldo Créd.": ["saldo crédito", "restante crédito"],
    "Dias": ["días transcurridos", "días de deuda"],
}

# # Pre‑construye las cadenas "principal (alias1, alias2, …)"
# SYN_STR = {k: f"{v[0]} ({', '.join(v[1:])})" for k, v in SYNONYMS.items()}

# TEMPLATE = (
#     f"{SYN_STR['Cliente']}: {{cliente}}. "
#     f"Tiene un {SYN_STR['Comprobante']} "
#     f"de {SYN_STR['Tipo']} «{{tipo}}», "
#     f"emitido el {SYN_STR['Fecha']} {{fecha}}, "
#     f"durante el {SYN_STR['Año - mes']} {{anio_mes}}."
# )

# # ──────────────────── GENERA DOCUMENTOS ────────────────
# docs, ids = [], []

# # itertuples() es mucho más rápido que iterrows()
# for i, row in enumerate(df.itertuples(index=False)):
#     # Por cómo funciona itertuples, los nombres con espacios se transforman:
#     # "Año - mes" → Año___mes  (espacio y guion → guiones bajos)
#     # Ajustamos con getattr():
#     docs.append(
#         Document(
#             page_content=TEMPLATE.format(
#                 cliente   = getattr(row, "Cliente",        ""),
#                 tipo      = getattr(row, "Tipo",           ""),
#                 fecha     = getattr(row, "Fecha",          ""),
#                 comp      = getattr(row, "Comprobante",    ""),
#                 anio_mes  = getattr(row, "Año___mes",      "")
#             ),
#             metadata={"id": str(i)}
#         )
#     )
#     ids.append(str(i))

# print(f"Total documentos generados: {len(docs)}")


# # Añade los documentos a Chroma en lotes con feedback
# for start in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embebiendo"):
#     end = start + BATCH_SIZE
#     vector_store.add_documents(
#         documents = docs[start:end],
#         ids       = ids[start:end]
#     )

# print("Indexación completa.")

# # ──────────────────── (Opcional) CONFIGURA RETRIEVER ───
# retriever = vector_store.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": len(docs), "score_threshold": 0.6}
# )

documents, ids = [], []

for i, row in df.iterrows():
    get = lambda col: str(row.get(col, "")).strip()

    content = (
        f"{SYNONYMS['Razón Social'][0]} ({', '.join(SYNONYMS['Razón Social'][1:])}): {get('Razón Social')}. "
        f"{SYNONYMS['Cód. Cliente'][0]} ({', '.join(SYNONYMS['Cód. Cliente'][1:])}): {get('Cód. Cliente')}. "
        f"Tiene un {SYNONYMS['Tipo Comprob. Créd.'][0]} ({', '.join(SYNONYMS['Tipo Comprob. Créd.'][1:])}) "
        f"«{get('Tipo Comprob. Créd.')}» con {SYNONYMS['Comprobante Créd.'][0]} «{get('Comprobante Créd.')}», "
        f"emitido el {SYNONYMS['Fecha Emisión Créd.'][0]} {get('Fecha Emisión Créd.')}. "

        f"Relacionado con un {SYNONYMS['Tipo Comprob. Déb.'][0]} «{get('Tipo Comprob. Déb.')}» "
        f"y {SYNONYMS['Comprobante Déb.'][0]} «{get('Comprobante Déb.')}» emitido el {get('Fecha Emisión Déb.')}."

        f" Vence (crédito) el {get('Fecha Vto. Créd.')}, "
        f"vence (débito) el {get('Fecha Vto. Déb.')}."

        f" Monto crédito: {get('Importe Créd.')}, "
        f"aplicado: {get('Importe Aplicado')}, "
        f"saldo: {get('Saldo Créd.')}, "
        f"días: {get('Dias')}. "

        f"Periodo (año-mes): {get('Año - mes')}."
    )

    # Crea el documento para tu índice/vector store
    doc = Document(
        page_content=content.strip(),
        metadata={"id": str(i)},
    )
    documents.append(doc)
    ids.append(str(i))

    # Batching
    MAX_BATCH_SIZE = 5000
    total = len(documents)
    print(f"Agregando {total} documentos en batches de {MAX_BATCH_SIZE}...")

    for i in range(0, total, MAX_BATCH_SIZE):
        batch_docs = documents[i:i + MAX_BATCH_SIZE]
        batch_ids = ids[i:i + MAX_BATCH_SIZE]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"Batch agregado: {i} al {min(i + MAX_BATCH_SIZE, total)}")

    print(f"{total} documentos agregados a la base.")

if add_documents:
   vector_store.add_documents(documents=documents, ids=ids)
   print(f"{len(documents)} documentos agregados a la base.")


# MAX_BATCH_SIZE = 5000

# documents, ids = [], []

# for i, row in df.iterrows():
#     # ----- genera el texto semántico -----
#     content = (
#         f"Cód. Cliente (comprador, destinatario, cliente): {row['Cód. Cliente']} "
#         f"Tiene un comprobante de Créd (número de comprobante, comprobante de credito, crédito, comprobante crédito): {row['Comprobante Créd.']} "
#         f"de tipo (tipo de comprobante de Créd, comprobante de créd, comprobante de credito) {row['Tipo Comprob. Créd.']} "
#         f"emitido en la fecha de emisión (fecha de emision de credito, fecha de emisión de crédito) {row['Fecha Emisión Créd.']} "
#         f"dentro del período (año‑mes, mes contable) {row['Año - mes']}."
#     )

#     print("Test generacion")
#     documents.append(Document(page_content=content, metadata={"id": str(i)}))
#     ids.append(str(i))

# # ----- una sola pasada de batches -----
# for start in range(0, len(documents), MAX_BATCH_SIZE):
#     print("Test cargado")
#     end = start + MAX_BATCH_SIZE
#     vector_store.add_documents(
#         documents=documents[start:end],
#         ids=ids[start:end]
#     )
#     print(f"Batch agregado: {start}–{end}")


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

    print(content)
    result = chain.invoke({"content": content, "question": quest})
    print(result)