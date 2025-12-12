import os
from ultralytics import YOLO

DATA_CONFIG = './yolo_config.yaml'
MODEL_BASE = 'yolov8n.pt'
PROJECT_NAME = 'yolo_pose_risk'
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 4

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def treinar_yolo():

    model = YOLO(MODEL_BASE)

    print(f"Iniciando treinamento!")

    try:
        results = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project='runs/detect',
            name=PROJECT_NAME,
            device=0
        )

        print("Treinamento concluído com sucesso!")

    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")

if __name__ == "__main__":
    treinar_yolo()