import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

res_testes = (480, 854)
res_display = (640,480)
res_yolo = (640,640)

mp_pose = mp.solutions.pose

TEXT = [
    "Sistema iniciado",
    "Aguardando postura",
    "Postura incorreta",
    "Ajuste os ombros"
]

ALLOWED_POINTS = [
    {
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW
    },
    {
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW
    }
]

ALLOWED_CONECTIONS = [
    {
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW)
    },
    {
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    }
]

def draw_text(img):
    x = 20
    y = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_spacing = 30
    scale = 0.7
    thickness = 2
    text_color = (0, 0, 255)
    h, w, _ = img.shape

    for i, text in enumerate(TEXT):
        y_pos = y + i * line_spacing

        cv2.putText(
            img,
            text,
            (x, y_pos),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )


def draw_allowed_connections(img, landmarks, conections):
    color = (0, 255, 0)
    thickness = 2
    h, w, _ = img.shape

    for p1, p2 in conections:
        lm1 = landmarks[p1.value]
        lm2 = landmarks[p2.value]

        x1, y1 = int(lm1.x * w), int(lm1.y * h)
        x2, y2 = int(lm2.x * w), int(lm2.y * h)

        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_allowed_points(img, landmarks, points):
    color = (255, 0, 0)
    radius_size = 5
    h, w, _ = img.shape

    for p in points:
        lm = landmarks[p.value]
        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(img, (x, y), radius_size, color, -1)


def main():
    points_selector = 1

    video_path = "data/validate/videoFrontal.mp4"

    cap = cv2.VideoCapture(video_path)

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            frame1 = cv2.resize(frame, res_testes)

            # Converte para RGB (Formato do MediaPipe)
            image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            # Protecao dos dados e otimizacao
            image.flags.writeable = False

            # Processamento da imagem
            results = pose.process(image)

            image.flags.writeable = True
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extrai as marcações
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass

            draw_text(image_bgr)

            # Desenha os landmarks
            if results.pose_landmarks:
                draw_allowed_points(image_bgr, landmarks, ALLOWED_POINTS[points_selector])
                draw_allowed_connections(image_bgr, landmarks, ALLOWED_CONECTIONS[points_selector])


            # Exibe o frame
            cv2.imshow('Pose Tracking (Pressione Q para sair)', image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if cv2.waitKey(1) & 0xFF == ord('1'):
                points_selector = 0

            if cv2.waitKey(1) & 0xFF == ord('2'):
                points_selector = 1


        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()