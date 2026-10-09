from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data="data.yaml",
        epochs=100,
        patience=100,   # sin parada anticipada: corre las 100 épocas
        imgsz=640,
        batch=16,
        device=0,
        name="entrenamiento_100_epocas",
    )

if __name__ == "__main__":
    main()