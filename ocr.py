import fitz  # PyMuPDF
import easyocr
import sys
from PIL import Image
import io
from tqdm import tqdm
import time

#file_path = sys.argv[1]
file_path = "C:/Users/BOTSMS/Desktop/INPUTS/CARBON HOYTS ABASTO 1025.pdf"

print("TEST")
print(file_path)

with open("resultado_ocr.txt", "w", encoding="utf-8"):
    pass

reader = easyocr.Reader(['es'])

doc = fitz.open(file_path)

for i in range(10):
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
                archivo.write(line[1] + "\n")
