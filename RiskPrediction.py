from operator import truediv

import cv2
import mediapipe as mp
import numpy as np

res_testes = (480, 854)
res_display = (640,480)
res_yolo = (640,640)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
points_selector = 0

"""
    Define os thresholeds para a identificação da etapa do salto
"""
JUMPING_TRESHOLD = 0.006
LANDING_TRESHOLD = 0.004

"""
    Texto que marca o instância do pulo (Saltando | Aterrisando | No chão)
"""
JUMPING_TEXT = ""

"""
    Marca quais dos movimentos de riscos já foram detectados
"""
RISK_DETECTED = {
    "stance_width" : False,
    "foot_landing" : False,
    "lateral_trunk" : False,
    "knee_flexion" : False,
    "trunk_flexion" : False,
    "ankle_plantar" : False
}

"""
    Array de textos dos riscos que já foram enconttrados
"""
TEXT = []

"""
Organização dos Pontos para cada avaliação de Risco:

Frontal:
 - Posição 1 -> Stance Width 
 - Posição 2 -> Foot landing
 - Posição 3 -> Lateral trunk Flexion

Lateral:
 - Posição 4 -> Knee Flexion 
 - Posição 5 -> Trunk Flexion
 - Posição 6 -> Ankle Plantar Flexion

"""
ALLOWED_POINTS = [
    {
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE

    },
    {
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    },
    {
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    }
]

ALLOWED_CONECTIONS = [
    {
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE)
    },
    {
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP),
    },
    {}
]
"""
    Funções para GUI
"""
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

        cv2.putText(
            img,
            JUMPING_TEXT,
            (x, h - 20),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

def draw_allowed_connections(img, landmarks, conections):
    color = (255, 255, 255)
    thickness = 2
    h, w, _ = img.shape

    for p1, p2 in conections:
        lm1 = landmarks[p1.value]
        lm2 = landmarks[p2.value]

        x1, y1 = int(lm1.x * w), int(lm1.y * h)
        x2, y2 = int(lm2.x * w), int(lm2.y * h)

        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_allowed_points(img, landmarks, points):
    color = (0, 0, 255)
    radius_size = 5
    h, w, _ = img.shape

    for p in points:
        lm = landmarks[p.value]
        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(img, (x, y), radius_size, color, -1)

"""
    FUNÇÕES AUXILIARES:
"""

"""
 Recebe dois pontos A e B e retorna o vetor AB
"""
def create_vector(A, B):
    return np.array(B) - np.array(A)

"""
 Calcula a distancia entre os pontos A e B
"""
def dist_euclidiana(A, B):
    return np.linalg.norm(create_vector(A,B))

"""
 Calcula o ponto médio entre dois pontos
"""
def mid_point(A, B):
    A = np.array(A)
    B = np.array(B)

    return np.divide(np.add(A, B), 2)

"""
 Calcula o angulo (em Grau) entre dois vetores usando o arccos dos vetores A e B
    Ang = arccos(A.B/|A|x|B|)
"""
def calculate_angle_arrays(A, B):
    # Norma dos vetores
    mod_A = np.linalg.norm(A)
    mod_B = np.linalg.norm(B)

    # Produto escalar
    dot_AB = np.dot(A, B)

    # Calculo do angulo em radiano
    rdn = np.arccos(dot_AB / (mod_A * mod_B))

    # Conversão para grau
    return np.degrees(rdn)

"""
 Calculo do ponto médio entre os calcanhar
"""
def foot_position(landmarks):
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    return (left_ankle.y + right_ankle.y) / 2

"""
    CÁLCULO DOS RISCOS:
"""

"""
• Distância dos pés:
 Cálculo da razão entre a distância dos ombros e a distância dos tornozelos
"""
def stance_width(landmarks, frame_num):
    # Coleta as referencias de cada ponto (Ombros e tornozelos)
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

    # Calcula as distâncias
    dist_shoulders = dist_euclidiana(left_shoulder, right_shoulder)
    dist_ankles = dist_euclidiana(left_ankle, right_ankle)

    # Calcula a razão
    ratio = dist_ankles / dist_shoulders

    # Caso n tenha sido detectado adiciona o texto na tela
    if not RISK_DETECTED["stance_width"]:
        if ratio < 0.8:
            TEXT.append("Narrow Stance: " + f"{ratio:.2f}" + " at frame:" + str(frame_num))
            RISK_DETECTED["stance_width"] = True
        elif ratio > 1.2:
            TEXT.append("Wide Stance: " + f"{ratio:.2f}" + ", at frame: " + str(frame_num))
            RISK_DETECTED["stance_width"] = True

"""
• Flexão Lateral do Tronco:
 Cálculo do angulo do vetor do tronco e um vetor perpendicular ao chão
"""
def risk_lateral_trunk(img, landmarks, frame_num):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_spacing = 30
    scale = 0.7
    thickness = 2
    text_color = (0, 255, 255)
    h, w, _ = img.shape

    # Coleta as referencias de cada ponto (Ombros e quadril)
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]

    # Mediana do ombro e quadril
    mid_shoulder = mid_point(left_shoulder, right_shoulder)
    mid_hip = mid_point(left_hip, right_hip)

    # Referencia central do topo
    mid_top = np.array((mid_hip[0], 0))

    # Vetores usado como base de referencia e do quadril
    base_arr = create_vector(mid_top, mid_hip)
    trunk_arr = create_vector(mid_shoulder, mid_hip)

    angle = calculate_angle_arrays(base_arr, trunk_arr)

    if not RISK_DETECTED["lateral_trunk"]:
        if angle > 7:
            RISK_DETECTED["lateral_trunk"] = True
            TEXT.append("Lateral Trunk detected at frame: " + str(frame_num))

    # Adiciona os vetores na imagem

    if points_selector == 1:
        # Conversão para pixels
        mid_shoulder_px = (int(mid_shoulder[0] * w), int(mid_shoulder[1] * h))
        mid_hip_px = (int(mid_hip[0] * w), int(mid_hip[1] * h))
        mid_top_px = (int(mid_top[0] * w), int(mid_top[1] * h))

        cv2.putText(
            img,
            f"{angle:.2f}",
            mid_hip_px,
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

        # Desenha os vetores
        cv2.line(img, mid_top_px, mid_hip_px, (255,255,255), 2)  # Base
        cv2.line(img, mid_shoulder_px, mid_hip_px, (0,0,255), 2)  # Tronco

"""
• Simterisa dos pés na aterrisagem:
 Cálculo da diferenca de altura das pontas dos pés na aterrisagem
"""
def risk_feet_symmetry(landmarks, frame_num):
    global JUMPING_TEXT

    if JUMPING_TEXT == "Aterrissando!":

        # Coleta as referencias dos pontos da ponta dos pés
        left_foot_height = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
        right_foot_height  = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y

        foot_assimetry = abs(left_foot_height - right_foot_height)

        if not RISK_DETECTED["foot_landing"]:
            if foot_assimetry > 0.05:
                RISK_DETECTED["foot_landing"] = True
                TEXT.append("Foot land assimetry at frame: " + str(frame_num))



def detect_risks(img, landmarks, frame_num):
    stance_width(landmarks, frame_num)
    risk_lateral_trunk(img, landmarks, frame_num)
    risk_feet_symmetry(landmarks, frame_num)

def main():
    global  JUMPING_TEXT
    global points_selector

    frame_num = 0
    past_frame_foot_y = None
    jump_vel = 0

    video_path_front = "data/validate/VideoFrontal.mp4"
    video_path_side = "data/validate/VideoLateral.mp4"

    cap_front = cv2.VideoCapture(video_path_front)
    cap_side = cv2.VideoCapture(video_path_side)

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose_front, mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose_side:
        while cap_front.isOpened() or cap_side.isOpened():
            frame_num+=1

            ret_front, frame_front = cap_front.read()
            ret_side, frame_side = cap_side.read()

            frame_front_resized = cv2.resize(frame_front, res_testes)
            frame_side_resized = cv2.resize(frame_side, res_testes)

            # Converte para RGB (Formato do MediaPipe)
            image_front = cv2.cvtColor(frame_front_resized, cv2.COLOR_BGR2RGB)
            image_side = cv2.cvtColor(frame_side_resized, cv2.COLOR_BGR2RGB)

            # Protecao dos dados e otimizacao
            image_front.flags.writeable = False
            image_side.flags.writeable = False

            # Processamento da imagem
            results_front = pose_front.process(image_front)
            results_side = pose_side.process(image_side)
            

            image_front.flags.writeable = True
            image_side.flags.writeable = True
            image_bgr_front = cv2.cvtColor(image_front, cv2.COLOR_RGB2BGR)
            image_bgr_side = cv2.cvtColor(image_side, cv2.COLOR_RGB2BGR)

            # Extrai as marcações
            lm_front = results_front.pose_landmarks.landmark
            lm_side = results_side.pose_landmarks.landmark

            if not lm_front or not lm_side:
                continue

            # Calculo para checar se esta na ação de pulo
            foot_y = foot_position(lm_front)

            # Calculo da variação da altura de Y (Identificar os estagios do pulo)
            if past_frame_foot_y:
                jump_vel = foot_y - past_frame_foot_y
            else:
                past_frame_foot_y = foot_y

            if np.abs(jump_vel) >= JUMPING_TRESHOLD:
                if jump_vel < 0:
                    JUMPING_TEXT = "Saltando!"
                else:
                    JUMPING_TEXT = "Aterrissando!"
            else:
                JUMPING_TEXT = "No chao"

            past_frame_foot_y = foot_y

            draw_text(image_bgr_front)

            # Desenha os landmarks
            draw_allowed_points(image_bgr_front, lm_front, ALLOWED_POINTS[points_selector])
            draw_allowed_connections(image_bgr_front, lm_front, ALLOWED_CONECTIONS[points_selector])
            draw_allowed_points(image_bgr_side, lm_side, ALLOWED_POINTS[points_selector])
            draw_allowed_connections(image_bgr_side, lm_side, ALLOWED_CONECTIONS[points_selector])

            detect_risks(image_bgr_front, lm_front, frame_num)

            # Concatena as duas imagens
            frame_concat = cv2.hconcat([image_bgr_front, image_bgr_side])

            cv2.imshow('Pose Tracking (Pressione Q para sair)', frame_concat)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('1'):
                points_selector = 0
            elif key == ord('2'):
                points_selector = 1
            elif key == ord('3'):
                points_selector = 2
            elif key == ord('q'):
                break


        cap_front.release()
        cap_side.release()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()