from ultralytics import YOLO

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