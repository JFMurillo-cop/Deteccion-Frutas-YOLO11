from pathlib import Path
from collections import Counter
import re

base = Path("datos/archive/fruit")

for split in ("train", "valid"):
    imgs = len(list((base / split / "images").glob("*.jpg")))
    labs = list((base / split / "labels").glob("*.txt"))
    print(f"\n== {split}: {imgs} imágenes, {len(labs)} etiquetas ==")

    archivos = Counter()   # prefijo -> cantidad de archivos
    cajas = Counter()      # (prefijo, clase) -> cantidad de cajas
    raras = 0

    for f in labs:
        prefijo = re.sub(r"\d+$", "", f.stem)
        archivos[prefijo] += 1
        for linea in f.read_text().splitlines():
            partes = linea.split()
            if len(partes) == 5:
                cajas[(prefijo, int(partes[0]))] += 1
            elif partes:
                raras += 1

    print("Archivos por prefijo:", dict(archivos))
    for (prefijo, clase), n in sorted(cajas.items()):
        print(f"  prefijo={prefijo!r}  clase={clase}  cajas={n}")
    print("Líneas que no tienen 5 valores:", raras)