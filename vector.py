from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

#aca va la lectura del documento
#archivo = "C:/Users/bmorales/Downloads/BD-Adm&Finanzas-DS_Backup (1).xlsx"
archivo = "C:/Users/bmorales/OneDrive - rmrconsultores.com/Escritorio/BD-Adm&Finanzas-DS_Backup (1).csv"
#df = pd.read_excel(archivo)
df = pd.read_csv(archivo, encoding='latin1')
print(df)

#embeddings = OllamaEmbeddings(model="mxbai_embed_large")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Combinar varias columnas como texto
        content = f"""
        A¤o - mes: {row.get('A¤o - mes', '')}
        C¢d. Cliente: {row.get('C¢d. Cliente', '')}
        Raz¢n Social: {row.get('Raz¢n Social', '')}
        Fecha Emisi¢n Cr‚d.: {row.get('Fecha Emisi¢n Cr‚d.', '')}
        Tipo Comprob. Cr‚d.: {row.get('Tipo Comprob. Cr‚d.', '')}
        """

        doc = Document(
            page_content=content.strip(),
            metadata={"id": str(i)},
        )

        documents.append(doc)
        ids.append(str(i))


vector_store = Chroma(
    collection_name="Testing",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"{len(documents)} documentos agregados a la base.")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)