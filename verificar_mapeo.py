from pathlib import Path
import cv2
import numpy as np

base = Path("datos/archive/fruit/train")
ANCHO, ALTO = 400, 300
muestras = {}  # clase -> (ruta_imagen, lista_de_cajas)

for lab in sorted((base / "labels").glob("*.txt")):
    cajas_por_clase = {}
    for linea in lab.read_text().splitlines():
        p = linea.split()
        if len(p) == 5:
            cajas_por_clase.setdefault(int(p[0]), []).append(list(map(float, p[1:])))
    for clase, cajas in cajas_por_clase.items():
        # preferimos imágenes donde SOLO aparece esa clase, para no confundirnos
        if clase not in muestras and len(cajas_por_clase) == 1:
            img = base / "images" / (lab.stem + ".jpg")
            if img.exists():
                muestras[clase] = (img, cajas)

paneles = []
for clase in range(11):
    if clase not in muestras:
        panel = np.zeros((ALTO, ANCHO, 3), np.uint8)
        cv2.putText(panel, f"clase {clase}: sin muestra", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        paneles.append(panel)
        continue
    ruta, cajas = muestras[clase]
    img = cv2.imread(str(ruta))
    if img is None:
        continue
    h, w = img.shape[:2]
    for cx, cy, bw, bh in cajas:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, w // 200))
    img = cv2.resize(img, (ANCHO, ALTO))
    cv2.rectangle(img, (0, 0), (ANCHO, 38), (0, 0, 0), -1)
    cv2.putText(img, f"CLASE {clase}  ({ruta.name})", (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    paneles.append(img)

while len(paneles) % 4:
    paneles.append(np.zeros((ALTO, ANCHO, 3), np.uint8))
filas = [np.hstack(paneles[i:i + 4]) for i in range(0, len(paneles), 4)]
cv2.imwrite("mapeo.jpg", np.vstack(filas))
print("Listo: se creó mapeo.jpg")