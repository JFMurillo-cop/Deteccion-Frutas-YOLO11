# Explicación del código

Este documento explica, archivo por archivo, qué hace el código del proyecto, para qué sirvió y qué resultados dio. Los fragmentos son las versiones usadas en el proyecto; si modificas un archivo, conviene actualizar aquí su explicación.

## 1. Visión general

El proyecto entrena un modelo que, dada una imagen, dibuja una caja alrededor de cada fruta y dice de qué fruta se trata. Esa tarea se llama **detección de objetos**.

El flujo es este:

```
Dataset de Kaggle ──► revisarclases.py ──► verificar_mapeo.py ──► data.yaml
                                                                     │
                                                                     ▼
 predecir.py ◄── best.pt ◄── entrenar.py ──► evaluar.py ──► metricas_por_clase.csv
```

| Archivo | Para qué sirvió |
|---|---|
| `revisarclases.py` | Comprobar que el dataset está completo y averiguar qué número de clase corresponde a cada fruta |
| `verificar_mapeo.py` | Confirmar ese mapeo mirando imágenes con sus cajas dibujadas |
| `data.yaml` | Decirle a YOLO dónde están las imágenes y cómo se llama cada clase |
| `entrenar.py` | Entrenar el modelo |
| `evaluar.py` | Medir el modelo con números y guardarlos en un archivo |
| `predecir.py` | Probar el modelo con fotos nuevas |
| `requirements.txt` | Listar los paquetes exactos usados, para poder reproducir el entorno |

## 2. Conceptos básicos

| Término | Qué significa |
|---|---|
| **CNN** | Red neuronal convolucional: un tipo de red que aprende a reconocer patrones en imágenes (bordes, texturas, formas) |
| **YOLO** | Familia de modelos de detección que predice cajas y clases de una sola pasada por la imagen. Aquí se usa YOLO11n, la versión más pequeña ("n" de *nano*) |
| **Detección de objetos** | Para cada objeto de la imagen, el modelo devuelve una caja, una clase y una confianza |
| **Transfer learning** | Partir de un modelo ya entrenado con otras imágenes (`yolo11n.pt`) y ajustarlo a las clases propias, en lugar de entrenar desde cero |
| **Época** | Una pasada completa por todas las imágenes de entrenamiento |
| **Lote (*batch*)** | Cantidad de imágenes que el modelo procesa a la vez antes de actualizar sus pesos |
| **Confianza (*conf*)** | Número entre 0 y 1 con el que el modelo expresa qué tan seguro está de una detección |
| **Precisión** | De todas las cajas que el modelo dibujó, qué fracción era correcta |
| **Recall** | De todas las frutas que realmente había, qué fracción encontró el modelo |
| **mAP50** | Promedio de la precisión media por clase, contando una detección como correcta si su caja se solapa al menos un 50 % con la caja real |
| **mAP50-95** | Lo mismo, pero promediando exigencias de solapamiento del 50 % al 95 %. Es más estricta y premia las cajas bien ajustadas |

## 3. `revisarclases.py`

**Para qué sirvió.** El dataset descargado trae las etiquetas como números (0 a 10), sin un archivo que diga qué fruta es cada uno. Este script cuenta cuántas imágenes, etiquetas y cajas hay, y las agrupa por el prefijo del nombre de archivo (por ejemplo `pine`, `grape`, `lemon`), que da pistas sobre la fruta.

```python
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
```

**Qué hace, paso a paso**

1. Recorre las carpetas `train` y `valid`.
2. Cuenta las imágenes `.jpg` y los archivos `.txt` de etiquetas.
3. Calcula el prefijo de cada etiqueta quitando los dígitos finales del nombre (`pine92` pasa a `pine`; `0002` pasa a una cadena vacía).
4. Lee cada línea de cada etiqueta. Una caja válida en formato YOLO tiene **5 valores**: clase, centro x, centro y, ancho y alto. Si una línea tiene otra cantidad, la cuenta como "rara".
5. Suma cuántas cajas hay de cada clase para cada prefijo, e imprime el resultado.

**Qué dio**

- `train`: 3845 imágenes y 3845 etiquetas. `valid`: 640 y 640. Cada imagen tiene su etiqueta.
- 0 líneas con formato raro en ambos conjuntos, es decir, todas las etiquetas tienen formato de caja.
- Once números de clase distintos (0 a 10).
- Los prefijos dieron pistas: `grape` siempre usa la clase 6, `lemon` la 5 y `pine` la 7.

Con esos conteos se calculó el total de cajas por clase y se comparó con las cifras publicadas por Dataset Ninja. Esa comparación llevó a la tabla de clases (apartado 5). Es una **deducción**, por eso se confirmó con el siguiente script.

## 4. `verificar_mapeo.py`

**Para qué sirvió.** Comprobar con los ojos que cada número de clase corresponde a la fruta que se había deducido.

```python
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
```

**Qué hace, paso a paso**

1. Recorre las etiquetas de `train` y guarda, para cada clase, la **primera imagen en la que solo aparece esa clase** (así no hay duda de qué fruta es).
2. Para cada clase del 0 al 10, abre la imagen y convierte las cajas del formato YOLO (valores entre 0 y 1) a píxeles: el centro y el tamaño se transforman en las esquinas de un rectángulo.
3. Dibuja los rectángulos en verde, reduce la imagen a 400×300 y le pone arriba el número de clase.
4. Junta los 11 paneles en una cuadrícula de 4 columnas y la guarda como `mapeo.jpg`.

**Qué dio.** Un mosaico de 1600×900 píxeles con las 11 clases:

| Clase | Fruta vista |
|---|---|
| 0 | manzana roja |
| 1 | mandarinas en una bandeja |
| 2 | pera |
| 3 | sandía |
| 4 | durián |
| 5 | limones |
| 6 | racimo de uvas oscuras |
| 7 | piña |
| 8 | pitahaya |
| 9 | melones amarillos |
| 10 | melón cantaloup |

Esto coincidió con la deducción y permitió escribir los nombres en `data.yaml`. Solo se revisó una imagen por clase, así que no demuestra que todas las etiquetas del dataset sean correctas.

## 5. `data.yaml`

**Para qué sirvió.** Es el archivo de configuración que YOLO lee para saber dónde están los datos y cómo se llaman las clases.

```yaml
path: C:/Users/juanf/Proyecto - Frutas/datos/archive/fruit
train: train/images
val: valid/images

names:
  0: manzana
  1: mandarina
  2: pera
  3: sandia
  4: durian
  5: limon
  6: uva
  7: pina
  8: pitahaya
  9: melon_coreano
  10: melon_cantaloup
```

| Campo | Qué significa |
|---|---|
| `path` | Carpeta raíz del dataset. Es una ruta absoluta del equipo donde se desarrolló y **hay que cambiarla en cada equipo** |
| `train` | Carpeta de imágenes de entrenamiento, relativa a `path` |
| `val` | Carpeta de imágenes de validación, relativa a `path` |
| `names` | Diccionario de número de clase a nombre |

YOLO encuentra la etiqueta de cada imagen reemplazando `/images/` por `/labels/` en su ruta y cambiando la extensión a `.txt`. Por eso la estructura `train/images` y `train/labels` es obligatoria. Los nombres van sin tildes ni eñes para evitar problemas de codificación en Windows.

## 6. `entrenar.py`

**Para qué sirvió.** Entrenar el modelo de detección con transfer learning.

```python
from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data="data.yaml",
        epochs=100,
        patience=100,
        imgsz=640,
        batch=16,
        device=0,
        name="entrenamiento_100_epocas",
    )

if __name__ == "__main__":
    main()
```

**Qué hace**

1. `YOLO("yolo11n.pt")` carga el modelo preentrenado (Ultralytics lo descarga la primera vez).
2. `model.train(...)` ajusta ese modelo a las 11 clases del dataset. Ultralytics reemplaza la capa final para que dé 11 clases en lugar de las 80 originales (el registro dice `Overriding model.yaml nc=80 with nc=11`).
3. En cada época, el modelo recorre las imágenes de entrenamiento y después se evalúa con las de validación. Al final conserva el mejor modelo.

| Parámetro | Valor | Qué hace |
|---|---|---|
| `data` | `data.yaml` | Archivo con rutas y clases |
| `epochs` | 100 | Cantidad de pasadas completas por el entrenamiento |
| `patience` | 100 | Épocas sin mejora antes de detenerse. Al ser igual a `epochs`, nunca corta antes de tiempo |
| `imgsz` | 640 | Las imágenes se redimensionan a 640 píxeles |
| `batch` | 16 | Imágenes por lote. Con el equipo de prueba, el entrenamiento usó unos 3.6 GB de memoria de la GPU |
| `device` | 0 | Usa la primera GPU |
| `name` | `entrenamiento_100_epocas` | Nombre de la carpeta de resultados en `runs/detect/` |

El bloque `if __name__ == "__main__":` es una práctica general en Windows para que los procesos auxiliares que cargan las imágenes no vuelvan a ejecutar el entrenamiento.

**Ajustes por defecto que aparecen en el registro** (no se cambiaron): optimizador `auto`, que eligió AdamW con tasa de aprendizaje de 0.000667; aumento de datos con mosaico, que se desactiva en las últimas 10 épocas (`close_mosaic=10`); volteo horizontal del 50 %; precisión mixta (`amp=True`); 8 procesos de carga (`workers=8`); `seed=0` y `deterministic=True`. Estos dos últimos explican que repetir el entrenamiento diera el mismo resultado.

**Qué dio**

- Duración: 0.739 horas (unos 45 minutos). En un entrenamiento anterior de 50 épocas, cada época tardó unos 21 a 23 segundos.
- Modelo de 2,591,985 parámetros y 6.5 GFLOPs, según el resumen que imprime YOLO.
- Archivos en `runs/detect/entrenamiento_100_epocas/`: los pesos en `weights/` (`best.pt` y `last.pt`, de unos 5.5 MB) y las gráficas que genera Ultralytics, como `labels.jpg`.
- Métricas finales de validación impresas en la terminal: precisión 0.91, recall 0.902, mAP50 0.962 y mAP50-95 0.791.
- Comparación: con 3 épocas se llegó a mAP50 0.869, con 50 a 0.962 y con 100 a 0.962. Pasar de 50 a 100 épocas casi no mejoró el resultado.

`best.pt` es el modelo de la época con mejor resultado en validación, y `last.pt` es el de la última época. Los demás scripts usan `best.pt`.

## 7. `evaluar.py`

**Para qué sirvió.** Medir el modelo final con un comando reproducible y guardar las métricas por clase en un archivo, para citarlas en el informe.

```python
import csv
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/entrenamiento_100_epocas/weights/best.pt")
    m = model.val(data="data.yaml", split="val", name="evaluacion_final", device=0)

    print(f"Precisión: {m.box.mp:.3f}  Recall: {m.box.mr:.3f}  "
          f"mAP50: {m.box.map50:.3f}  mAP50-95: {m.box.map:.3f}")

    with open("metricas_por_clase.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "precision", "recall", "mAP50", "mAP50-95"])
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            w.writerow([m.names[c], round(p, 3), round(r, 3), round(ap50, 3), round(ap, 3)])
    print("Guardado: metricas_por_clase.csv")

if __name__ == "__main__":
    main()
```

**Qué hace**

1. Carga `best.pt` y ejecuta `model.val(...)`, que evalúa el modelo sobre las imágenes de validación definidas en `data.yaml`.
2. `m.box.mp`, `m.box.mr`, `m.box.map50` y `m.box.map` son la precisión media, el recall medio, el mAP50 y el mAP50-95 de todas las clases.
3. Recorre las clases con `ap_class_index` y, para cada una, pide sus cuatro métricas con `class_result(i)`.
4. Escribe una fila por clase en `metricas_por_clase.csv`.

**Qué dio**

| Métrica | Valor |
|---|---|
| Precisión | 0.902 |
| Recall | 0.907 |
| mAP50 | 0.963 |
| mAP50-95 | 0.790 |

Evaluó 639 imágenes con 2427 objetos. La tabla completa por clase está en `README.md` y en `metricas_por_clase.csv`. Las clases con mejor mAP50-95 son pera (0.904), melón cantaloup (0.863), manzana (0.848) y melón coreano (0.848); las más bajas, pitahaya (0.647), uva (0.720) y piña (0.720).

Estas cifras difieren levemente de las impresas al final del entrenamiento (0.91, 0.902, 0.962 y 0.791). En la terminal, la validación del entrenamiento usó 20 lotes y `evaluar.py` 40; no se comprobó si eso explica la diferencia. El resultado se guarda además en `runs/detect/evaluacion_final/`.

## 8. `predecir.py`

**Para qué sirvió.** Ver cómo se comporta el modelo con fotos que no son del dataset.

```python
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/entrenamiento_100_epocas/weights/best.pt")
    model.predict(
        source="imagenes_prueba",
        conf=0.25,
        save=True,
        name="resultados_100_epocas",
        device=0,
    )

if __name__ == "__main__":
    main()
```

| Parámetro | Valor | Qué hace |
|---|---|---|
| `source` | `imagenes_prueba` | Carpeta con las fotos a analizar |
| `conf` | 0.25 | Solo se muestran las detecciones con confianza de al menos 0.25 |
| `save` | `True` | Guarda una copia de cada foto con las cajas y etiquetas dibujadas |
| `name` | `resultados_100_epocas` | Carpeta de salida en `runs/detect/` |
| `device` | 0 | Usa la GPU |

**Qué dio.** Se probaron 25 fotos (5 de cada una de cinco frutas) a unos 5 o 6 milisegundos por imagen:

| Fruta | ¿La conoce el modelo? | Resultado |
|---|---|---|
| mandarina (5) | Sí | Detectó mandarinas en 3 fotos; en `mandarina02` no detectó nada y en `mandarina03` vio 2 peras |
| manzana (5) | Sí | Detectó manzanas en 4 fotos; en `manzana05` vio 2 pitahayas |
| banano (5) | No | 1 detección falsa (un melón coreano) y 4 sin detecciones |
| cereza (5) | No | 5 con detecciones falsas (manzanas, uvas y una pitahaya) |
| fresa (5) | No | 5 con detecciones falsas (durianes, sandía, piñas, pitahayas) |

En resumen, con las frutas que conoce acertó en 7 de 10 fotos, y con las 15 de frutas desconocidas hubo 11 detecciones falsas y 4 fotos sin detecciones. Es un detector de clases fijas: no tiene una opción de "no sé qué es", así que ante una fruta nueva fuerza una de sus 11 clases. Estos resultados salen del texto de la terminal; no se revisaron manualmente las cajas. Son pocas imágenes, por lo que sirven como comprobación cualitativa y no como medida de rendimiento.

## 9. `requirements.txt`

Lista los 40 paquetes con la versión exacta usada. Se generó así:

```powershell
"--extra-index-url https://download.pytorch.org/whl/cu126" | Out-File -Encoding utf8 requirements.txt
pip freeze | Out-File -Encoding utf8 -Append requirements.txt
```

- La primera línea indica a `pip` que, además del repositorio normal, busque paquetes en el de PyTorch para CUDA 12.6. Hace falta porque `torch==2.14.1+cu126` y `torchvision==0.29.1+cu126` (el sufijo `+cu126` significa "con CUDA 12.6") no están en el repositorio normal.
- Se guarda en UTF-8 porque, en PowerShell, `pip freeze > requirements.txt` puede dejarlo en otra codificación que `pip` no lee bien.
- Los paquetes principales son `ultralytics` (YOLO), `torch` y `torchvision` (el motor de redes neuronales), `opencv-python` (leer y dibujar imágenes), `numpy`, `matplotlib` y `PyYAML`. El resto son dependencias de esos.

## 10. Carpeta `runs/detect/`

Ultralytics guarda aquí lo que genera cada ejecución. Cada subcarpeta lleva el nombre que se indicó en `name`.

| Carpeta | Generada por | Contenido principal |
|---|---|---|
| `entrenamiento_100_epocas/` | `entrenar.py` | Pesos del modelo en `weights/` y las gráficas del entrenamiento |
| `evaluacion_final/` | `evaluar.py` | Resultados de la evaluación sobre validación |
| `resultados_100_epocas/` | `predecir.py` | Las fotos de prueba con las cajas dibujadas |

Si una carpeta ya existe, Ultralytics crea otra con un número al final.

## 11. Problemas encontrados y cómo se resolvieron

| Problema | Qué pasó y cómo se resolvió |
|---|---|
| Dataset de segmentación en lugar de detección | El primer dataset elegido era de segmentación (polígonos). Se descartó y se buscó uno con cajas: Fruit Object Detection de Kaggle |
| PyTorch bloqueado por Windows | Al importar PyTorch apareció `DLL load failed ... Una directiva de Control de aplicaciones bloqueó este archivo`. La causa fue el Control Inteligente de Aplicaciones (Smart App Control) de Windows; el error desapareció al cambiar esa configuración. Es una decisión de seguridad que cada usuario debe valorar |
| PyTorch sin GPU | Al instalar `ultralytics` se instaló `torch` en su versión de CPU. Se reemplazó por la versión con CUDA 12.6, y `torch.cuda.is_available()` pasó a `True` |
| Entorno virtual sin Ultralytics | Ultralytics se había instalado en el Python general, no en el `venv`. Se reinstaló dentro del entorno |
| Imágenes "corruptas" | 9 imágenes de `train` y 1 de `valid` son GIF con extensión `.jpg`. Ultralytics las ignora sola |
| Clases sin nombre | El dataset solo trae números. Se dedujeron con `revisarclases.py` y se verificaron con `verificar_mapeo.py` |
| Carpeta de prueba con imágenes de más | `imagenes_prueba/` llegó a tener 792 archivos, porque se habían mezclado fotos numeradas del dataset con las 25 fotos propias. Se apartaron las numeradas para que la prueba use solo las 25 |

## 12. Limitaciones

- El modelo solo reconoce las 11 clases con las que se entrenó.
- No hay un conjunto de prueba independiente: la validación sirve tanto para elegir `best.pt` como para medir, así que las métricas pueden ser algo optimistas.
- La prueba con fotos nuevas es pequeña (25 fotos) y no se revisó visualmente.
- La licencia del dataset es desconocida, por eso no se incluye en el repositorio.
