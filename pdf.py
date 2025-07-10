import fitz  # PyMuPDF
import requests
import time

# === CONFIGURACIÓN ===
RUTA_PDF = "inputs/CARBON HOYTS ABASTO 1025.pdf"
TAMAÑO_CHUNK = 3000  # caracteres
MODELO_OLLAMA = "llama3"
PREGUNTA = "¿Cuál es el total a pagar?"

# === FUNCIONES ===

def extraer_texto_pdf(ruta_pdf):
    doc = fitz.open(ruta_pdf)
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()
    return texto

def dividir_en_chunks(texto, tamaño):
    return [texto[i:i + tamaño] for i in range(0, len(texto), tamaño)]

def consultar_ollama(chunk, pregunta):
    payload = {
        "model": MODELO_OLLAMA,
        "messages": [
            {"role": "system", "content": "Sos un asistente que analiza contenido de documentos PDF."},
            {"role": "user", "content": f"Contenido del documento:\n{chunk}"},
            {"role": "user", "content": pregunta}
        ]
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        print(f"Error al consultar Ollama: {e}")
        return ""

# === FLUJO PRINCIPAL ===

def main():
    print("Extrayendo texto del PDF...")
    texto = extraer_texto_pdf(RUTA_PDF)

    print("Dividiendo texto en partes...")
    partes = dividir_en_chunks(texto, TAMAÑO_CHUNK)

    respuestas = []
    for i, chunk in enumerate(partes):
        print(f"\n🔍 Analizando chunk {i+1}/{len(partes)}...")
        respuesta = consultar_ollama(chunk, PREGUNTA)
        respuestas.append(respuesta)
        time.sleep(1)  # pequeña pausa para no saturar

    print("\n=== RESPUESTAS PARCIALES ===")
    for i, r in enumerate(respuestas):
        print(f"\n--- Chunk {i+1} ---\n{r.strip()}")

    print("\n=== RESPUESTA CONSOLIDADA ===")
    resumen = "\n".join(respuestas)
    respuesta_final = consultar_ollama(resumen, f"Consolidá esta información y respondé: {PREGUNTA}")
    print(respuesta_final.strip())

if __name__ == "__main__":
    main()
