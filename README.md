# Detección de frutas con YOLO11n

Miniproyecto de redes neuronales convolucionales: detección de objetos (cajas delimitadoras) en imágenes de 11 clases de frutas, usando transfer learning sobre YOLO11n (Ultralytics) en Python.

**Autor:** JFMurillo-cop

> **Importante: antes de ejecutar nada hay que instalar varias cosas** (Python, Git, el controlador de la GPU y los paquetes de `requirements.txt`). Mira la sección *Qué hay que instalar* y luego *Instalación paso a paso*. Sin eso, los scripts no funcionan.

> **Importante: tendrás que cambiar las rutas.** En `data.yaml`, la línea `path:` apunta a una carpeta del equipo donde se hizo el proyecto (`C:/Users/juanf/...`). Cámbiala por la ruta de `datos/archive/fruit` en tu computador (usa `/` como separador). Además, ejecuta siempre los comandos **desde la carpeta raíz del proyecto**. Los pasos detallados están en *Instalación*, paso 6.

## Resumen

| Elemento | Detalle |
|---|---|
| Tarea | Detección de objetos |
| Modelo | YOLO11n preentrenado (`yolo11n.pt`), ajustado con transfer learning |
| Dataset | Fruit Object Detection (Kaggle, autor `eunpyohong`) |
| Clases | 11 |
| Imágenes | 3845 en `train` y 640 en `valid` (según los archivos descargados) |
| Entrenamiento | 100 épocas, imágenes de 640 px, lote de 16, 0.739 horas |
| Equipo de prueba | Windows 11, Python 3.13, GPU NVIDIA RTX 2070 SUPER (8 GB) |

Para entender qué hace cada archivo de código, qué produce y por qué, lee **[EXPLICACION_CODIGO.md](EXPLICACION_CODIGO.md)**.

## Qué incluye y qué no incluye este repositorio

**Incluye:** los scripts de Python, `data.yaml`, `requirements.txt`, `mapeo.jpg`, `metricas_por_clase.csv`, las 25 fotos de prueba (`imagenes_prueba/`), el modelo entrenado (`runs/detect/entrenamiento_100_epocas/weights/best.pt`, unos 5.5 MB) y los resultados y gráficas que generó Ultralytics en `runs/detect/`.

**No incluye:**

- **El dataset (`datos/`).** Pesa unos 2 GB y su licencia aparece como "Unknown" en Kaggle y Dataset Ninja. Hay que descargarlo de Kaggle (paso 5 de la instalación).
- **El entorno virtual (`venv/`).** Se recrea con `requirements.txt`.
- **Los modelos preentrenados** que Ultralytics descarga solo (`yolo11n.pt`, `yolo26n.pt`).

Las fotos de `imagenes_prueba/` vienen de internet y se usan solo con fines académicos.

## Qué hay que instalar

Instálalo en este orden. **Todo es necesario para que el proyecto funcione**, salvo lo marcado como opcional.

| # | Qué instalar | Para qué sirve | Dónde conseguirlo | Cómo comprobar que quedó bien |
|---|---|---|---|---|
| 1 | **Python 3.13** | Ejecutar todo el código | https://www.python.org/downloads/ | `python --version` |
| 2 | **Git** | Clonar este repositorio | https://git-scm.com/downloads | `git --version` |
| 3 | **Controlador de la GPU NVIDIA** | Que PyTorch pueda usar la GPU | https://www.nvidia.com/Download/index.aspx | `nvidia-smi` |
| 4 | **Paquetes de Python** (PyTorch, Ultralytics, OpenCV y otros) | Entrenar, evaluar y predecir | `pip install -r requirements.txt` (paso 3 de la instalación) | `python -c "import torch; print(torch.cuda.is_available())"` debe imprimir `True` |
| 5 | **Cuenta de Kaggle** | Descargar el dataset | https://www.kaggle.com | Poder iniciar sesión |
| 6 | VS Code *(opcional)* | Editor para abrir y ejecutar el proyecto | https://code.visualstudio.com | Abrir la carpeta del proyecto |

Notas:

- Al instalar Python en Windows, el instalador suele ofrecer una casilla para agregar Python al PATH (a veces llamada *Add python.exe to PATH*). Márcala; si no, el comando `python` no funcionará en la terminal.
- Si no tienes una GPU NVIDIA, el proyecto puede correr en CPU, pero el entrenamiento será mucho más lento y hay que cambiar `device=0` por `device="cpu"` en los scripts. Esta variante no se probó. Tampoco se probó con otras versiones de Python: el proyecto se desarrolló con la 3.13, y la guía de Ultralytics Academy indica que las versiones 3.9 a 3.12 están bien soportadas.
- **Sistema operativo:** el proyecto se probó solo en Windows 11.
- **Espacio en disco:** el zip del dataset pesa unos 2 GB según Kaggle, más el entorno virtual con PyTorch.
- **Internet:** hace falta para instalar los paquetes y porque Ultralytics descarga `yolo11n.pt` la primera vez que se entrena.

## Instalación paso a paso

Los comandos son para PowerShell. Ejecútalos en orden.

**1. Clonar el repositorio**

```powershell
git clone https://github.com/JFMurillo-cop/Deteccion-Frutas-YOLO11.git
cd Deteccion-Frutas-YOLO11
```

Si el repositorio tiene otro nombre, cambia la dirección por la de tu repositorio.

**2. Crear y activar el entorno virtual**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Si Windows bloquea el script de activación, ejecuta primero esto (solo vale para esa ventana de terminal) y vuelve a activar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Sabrás que está activo porque la línea empieza con `(venv)`.

**3. Instalar los paquetes**

```powershell
pip install -r requirements.txt
```

La primera línea de `requirements.txt` apunta al repositorio de PyTorch con CUDA 12.6, necesario para `torch==2.14.1+cu126`. Esta instalación no se probó en otro equipo. Tarda varios minutos porque PyTorch pesa mucho.

**4. Verificar que PyTorch usa la GPU**

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Debe imprimir algo como `2.14.1+cu126 True`. Si imprime `False`, mira la sección *Solución de problemas*.

**5. Descargar el dataset**

1. Entra a https://www.kaggle.com/datasets/eunpyohong/fruit-object-detection (puede pedir iniciar sesión).
2. Elige *Download dataset as zip* (unos 2 GB).
3. Crea una carpeta `datos` en la raíz del proyecto y descomprime el zip dentro.

Debe quedar esta estructura (el zip también trae dos archivos `.pth.tar` que no se usan):

```
datos/archive/fruit/
├── train/images   train/labels
└── valid/images   valid/labels
```

**6. Ajustar la ruta en `data.yaml`**

Abre `data.yaml` y cambia la línea `path:` por la ruta absoluta de `datos/archive/fruit` en tu equipo, usando `/` como separador. Ejemplo:

```yaml
path: C:/Users/tu_usuario/Deteccion-Frutas-YOLO11/datos/archive/fruit
```

## Orden de ejecución

Con el entorno activo `(venv)` y desde la raíz del proyecto:

| Paso | Comando | Qué produce | Qué debes ver |
|---|---|---|---|
| 1 | `python revisarclases.py` | Conteos por clase en la terminal | 3845 imágenes y 3845 etiquetas en `train`; 640 y 640 en `valid`; 0 líneas con formato raro |
| 2 | `python verificar_mapeo.py` | `mapeo.jpg` | Un mosaico con 11 paneles, una fruta por clase, con sus cajas |
| 3 | Revisar `data.yaml` | Nada | Ruta correcta y las 11 clases numeradas de 0 a 10 |
| 4 | `python entrenar.py` | `runs/detect/entrenamiento_100_epocas/` | Unos 45 minutos con una RTX 2070 SUPER; tabla de métricas al final |
| 5 | `python evaluar.py` | `metricas_por_clase.csv` y `runs/detect/evaluacion_final/` | Línea con precisión, recall, mAP50 y mAP50-95 |
| 6 | `python predecir.py` | `runs/detect/resultados_100_epocas/` | Una línea por imagen con lo detectado |

Notas:

- Si una carpeta de `runs/detect/` ya existe, Ultralytics crea otra con un número agregado (por ejemplo `resultados_100_epocas-2`). Para repetir con los mismos nombres, renombra o mueve `runs` antes.
- La primera vez que se entrena, Ultralytics descarga `yolo11n.pt` (hace falta internet).
- Mantén el equipo despierto durante el entrenamiento.
- Los pasos 1 y 2 no son necesarios para entrenar: documentan cómo se comprobó el dataset y cómo se dedujeron los nombres de las clases.

### Solo probar el modelo ya entrenado

Si solo quieres ver el modelo en acción, no hace falta descargar el dataset ni entrenar. Con el entorno instalado, pon tus fotos en `imagenes_prueba/` (puedes reemplazar las que trae) y ejecuta:

```powershell
python predecir.py
```

Usa el modelo incluido en `runs/detect/entrenamiento_100_epocas/weights/best.pt`.

## Dataset

- **Fuente:** https://www.kaggle.com/datasets/eunpyohong/fruit-object-detection
- **Licencia:** desconocida. Se usa solo con fines académicos y no se redistribuye.
- **Formato:** etiquetas YOLO (un `.txt` por imagen con `clase x_centro y_centro ancho alto`, valores normalizados).
- **Imágenes descartadas:** al entrenar, Ultralytics ignoró 9 imágenes de `train` y 1 de `valid` porque son GIF con extensión `.jpg`, y restauró 4 JPEG dañados. Se usaron 3836 imágenes de entrenamiento y 639 de validación.

### Clases

El dataset no incluye los nombres de las clases. Se dedujeron comparando el número de cajas por clase con las cifras publicadas por Dataset Ninja y se verificaron visualmente con `verificar_mapeo.py`.

| Índice | Clase |
|---|---|
| 0 | manzana |
| 1 | mandarina |
| 2 | pera |
| 3 | sandia |
| 4 | durian |
| 5 | limon |
| 6 | uva |
| 7 | pina |
| 8 | pitahaya |
| 9 | melon_coreano |
| 10 | melon_cantaloup |

Los nombres van sin tildes ni eñes a propósito, para evitar problemas de codificación en Windows.

## Estructura del proyecto

```
.
├── README.md                 # esta guía
├── EXPLICACION_CODIGO.md     # explicación detallada de cada archivo de código
├── requirements.txt          # paquetes y versiones usadas
├── data.yaml                 # rutas del dataset y nombres de las 11 clases
├── revisarclases.py          # cuenta cajas por clase y por prefijo de archivo
├── verificar_mapeo.py        # dibuja una imagen por clase (mapeo.jpg)
├── entrenar.py               # entrenamiento (100 épocas)
├── evaluar.py                # métricas finales y metricas_por_clase.csv
├── predecir.py               # predicciones sobre imagenes_prueba
├── mapeo.jpg                 # resultado de verificar_mapeo.py
├── metricas_por_clase.csv    # resultado de evaluar.py
├── imagenes_prueba/          # 25 fotos nuevas para probar el modelo
├── runs/detect/              # resultados generados por Ultralytics
└── datos/                    # dataset (NO incluido; se descarga de Kaggle)
```

## Resultados

### Entrenamiento (100 épocas)

El entrenamiento usó `seed=0` y `deterministic=True`. Al repetirlo desde cero se obtuvieron las mismas métricas finales (mAP50-95 de 0.791 en ambas corridas).

Un entrenamiento previo de 50 épocas dio un resultado casi igual (mAP50 0.962 y mAP50-95 0.789), lo que sugiere que el modelo ya se había estabilizado.

### Evaluación final (`evaluar.py`, 639 imágenes de validación, 2427 objetos)

| Métrica | Valor |
|---|---|
| Precisión | 0.902 |
| Recall | 0.907 |
| mAP50 | 0.963 |
| mAP50-95 | 0.790 |

| Clase | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| manzana | 0.900 | 0.925 | 0.975 | 0.848 |
| mandarina | 0.951 | 0.872 | 0.974 | 0.802 |
| pera | 0.918 | 0.981 | 0.991 | 0.904 |
| sandia | 0.747 | 0.895 | 0.916 | 0.785 |
| durian | 0.811 | 0.987 | 0.967 | 0.811 |
| limon | 0.862 | 0.867 | 0.926 | 0.742 |
| uva | 0.929 | 0.846 | 0.952 | 0.720 |
| pina | 0.957 | 0.897 | 0.971 | 0.720 |
| pitahaya | 0.925 | 0.931 | 0.985 | 0.647 |
| melon_coreano | 0.951 | 0.950 | 0.977 | 0.848 |
| melon_cantaloup | 0.970 | 0.828 | 0.957 | 0.863 |

Las cifras impresas al final del entrenamiento (precisión 0.910, recall 0.902, mAP50 0.962, mAP50-95 0.791) difieren levemente de las de `evaluar.py`. En la terminal, la validación del entrenamiento procesó las imágenes en 20 lotes y `evaluar.py` en 40; no se verificó si eso explica la diferencia. Aquí se usan las de `evaluar.py` porque se reproducen con un solo comando.

### Prueba con 25 fotos nuevas (`predecir.py`)

Se probaron 5 fotos de cada una de cinco frutas: dos que el modelo conoce (mandarina y manzana) y tres que no están entre sus clases (banano, cereza y fresa). Los resultados salen del texto que imprime la terminal; no se hizo una revisión manual de las cajas dibujadas.

| Grupo | Resultado |
|---|---|
| Frutas conocidas (10 fotos) | Fruta correcta detectada en 7. Fallos: `mandarina02` (sin detecciones), `mandarina03` (2 peras) y `manzana05` (2 pitahayas) |
| Frutas desconocidas (15 fotos) | 11 con detecciones falsas y 4 sin detecciones |

Es una comprobación cualitativa con pocas imágenes y no sirve para calcular un porcentaje de acierto fiable.

## Limitaciones

- **Clases fijas.** El modelo solo puede nombrar sus 11 clases. Ante frutas desconocidas (banano, cereza, fresa) asigna una clase equivocada o no detecta nada.
- **Sin conjunto de prueba.** El dataset solo trae `train` y `valid`, y `best.pt` se elige con la misma validación. Las métricas pueden ser algo optimistas respecto a imágenes totalmente nuevas.
- **Rendimiento fuera del estilo del dataset.** Las métricas de validación se refieren a imágenes parecidas a las de entrenamiento; con fotos de otro estilo el rendimiento baja.
- **Clases más difíciles** según mAP50-95: pitahaya (0.647), uva (0.720), piña (0.720) y limón (0.742). La pitahaya tiene solo 58 objetos en validación.
- **Licencia del dataset desconocida.** Se cita la fuente y no se redistribuyen los datos.

## Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| `ImportError: DLL load failed while importing _C` con el mensaje "Una directiva de Control de aplicaciones bloqueó este archivo" | Windows (Control Inteligente de Aplicaciones / Smart App Control) bloqueó una DLL de PyTorch. Es una decisión de seguridad de cada usuario; ver la sección de problemas de `EXPLICACION_CODIGO.md` |
| `torch.cuda.is_available()` da `False` | Se instaló PyTorch de CPU. Ejecuta `pip uninstall -y torch torchvision` y luego `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| `ModuleNotFoundError: No module named 'ultralytics'` | El entorno virtual no está activo o faltan paquetes. Ejecuta `.\venv\Scripts\Activate.ps1` y `pip install -r requirements.txt` |
| Error de "ejecución de scripts deshabilitada" al activar el entorno | Ejecuta `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` en esa terminal |
| Error de memoria de la GPU al entrenar | Baja `batch=16` a `batch=8` o `batch=4` en `entrenar.py` |
| Mensajes `Invalid image format GIF` | Es normal con este dataset: son 9 imágenes de `train` y 1 de `valid`, que se ignoran |
| Aparece una carpeta como `entrenamiento_100_epocas2` o `resultados_100_epocas-2` | La carpeta ya existía. Renombra o mueve `runs` antes de repetir |
| `predecir.py` procesa más imágenes de las esperadas | `imagenes_prueba/` contiene otras fotos. Debe tener solo las que quieras probar |

## Referencias

- Dataset: https://www.kaggle.com/datasets/eunpyohong/fruit-object-detection
- Estadísticas del dataset: https://datasetninja.com/fruit-object-detection
- Ultralytics YOLO: https://docs.ultralytics.com
