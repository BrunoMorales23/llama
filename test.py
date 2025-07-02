import fitz  # PyMuPDF
import easyocr
from PIL import Image
import io

with open("resultado_ocr.txt", "w", encoding="utf-8"):
    pass

# Crear lector EasyOCR (idioma español)
reader = easyocr.Reader(['es'])

# Abrir el PDF
#doc = fitz.open(".\CV Bruno Morales 2025 EN.pdf")
doc = fitz.open(".\FacturaTestPDF.pdf")

# Procesar cada página del PDF
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    # Guardar temporalmente si es necesario:
    img_path = f"pagina_{i+1}.png"
    img.save(img_path)

    # OCR con EasyOCR
    result = reader.readtext(img_path)

with open("resultado_ocr.txt", "w", encoding="utf-8") as archivo:
    print(f"--- Página {i+1} ---")
    for line in result:
        print(line[1])  # texto reconocido
        archivo.write(line[1] + "\n")
