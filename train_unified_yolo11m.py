from ultralytics import YOLO

def train():
    model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\train_unified\yolo11m_unified_safety\weights\best.pt") 
    model.train(
        data= r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\new_unify_safety\new_unify_safety.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        device=0,
        project="runs/fine_tune_unify_safety",
        name="finetune_unified_safety",
        lr0=0.001,
        resume=False,
    )

if __name__ == "__main__":
    train()
