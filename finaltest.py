import os
import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ──────────────────────── CONFIGURACIÓN ─────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"
archivo = "C:/Users/bmorales/OneDrive - rmrconsultores.com/Escritorio/llama/SHORTVER.csv"
db_location = "./chrome_langchain_db"
collection_name = "Testing"
embedding_model = "nomic-embed-text"
llm_model = "llama3.2"
MAX_BATCH_SIZE = 5000

# ──────────────────────── SINÓNIMOS ─────────────────────────────
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

# ──────────────────────── LECTURA DE DATOS ──────────────────────
try:
    df = pd.read_csv(archivo, sep=None, encoding="utf-8", index_col=False, engine="python")
    df.columns = df.columns.str.strip()
    print(df)
    print(f"Archivo cargado correctamente con {df.shape[0]} filas.")
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    exit()

# ──────────────────────── CONFIGURA EMBEDDINGS Y DB ─────────────
embeddings = OllamaEmbeddings(model=embedding_model)
vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=db_location,
    embedding_function=embeddings
)

add_documents = not os.path.exists(db_location)

# ──────────────────────── INDEXACIÓN ────────────────────────────
# if add_documents:
documents, ids = [], []

for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generando documentos"):
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

        documents.append(Document(page_content=content, metadata={"id": str(i)}))
        ids.append(str(i))

print(f"Agregando {len(documents)} documentos en lotes de {MAX_BATCH_SIZE}...")
for start in range(0, len(documents), MAX_BATCH_SIZE):
        end = start + MAX_BATCH_SIZE
        vector_store.add_documents(documents=documents[start:end], ids=ids[start:end])
        print(f"Batch agregado: {start}–{end}")
# else:
#     print("Base de datos ya existente. No se agregarán documentos.")

# ──────────────────────── CONFIGURA RETRIEVER Y LLM ─────────────
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": df.shape[0], "score_threshold": 0.65}
)

model = OllamaLLM(model=llm_model, temperature=0.1)

template = """Tu único rol es ser un asistente de recuperación de información.

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

# ──────────────────────── LOOP DE PREGUNTAS ─────────────────────
while True:
    print("\n--------------------------------")
    quest = input("Pregunta lo que quieras (X para salir): ").strip()
    if quest.upper() == "X":
        print("Saliendo...")
        break

    retrieved_docs = retriever.invoke(quest)
    print(retrieved_docs)
    if not retrieved_docs:
        content = "[]"
    else:
        content = "\n".join([doc.page_content for doc in retrieved_docs])

    result = chain.invoke({"content": content, "question": quest})
    print("\n🟩 Resultado:\n")
    print(result)
