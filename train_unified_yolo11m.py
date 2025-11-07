from ultralytics import YOLO

def train():
    model = YOLO("yolo11m.pt") 
    model.train(
        data= r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\unified_ppe_fire.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=8,
        project="runs/train_unified",
        name="yolo11m_unified_safety",
        cache=True,
        amp=True,
        lr0=0.005,
        optimizer="SGD",
        mosaic=0.5,
        mixup=0.1,
        patience=30
    )

if __name__ == "__main__":
    train()
