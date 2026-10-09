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