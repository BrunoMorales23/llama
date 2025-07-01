from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

#aca va la lectura del documento

embeddings = OllamaEmbeddings(model="mxbai_embed_large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in VARIABLEFORFILECONTENT.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={},
            id=str(i)
        )