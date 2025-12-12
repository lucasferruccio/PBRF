import os
import shutil

import cv2
import mediapipe as mp
import numpy as np

res_testes = (480, 854)
res_display = (640,480)
res_yolo = (640,640)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
points_selector = 0
knee_angle = 0

risk_detection_front = {}
risk_detection_side = {}

"""
    Define os thresholeds para a identificação da etapa do salto
"""
JUMPING_TRESHOLD = 0.006
LANDING_TRESHOLD = 0.004

"""
    Texto que marca o instância do pulo (Saltando | Aterrisando | No chão)
"""
JUMPING_TEXT = "No chao"

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
ALLOWED_POINTS_FRONT = [
    {
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE

    },
    {
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    },
    {
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    }
]

ALLOWED_CONECTIONS_FRONT = [
    {
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE)
    },
    {},
    {
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP),
    }
]

ALLOWED_POINTS_SIDE = [
    {
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE
    },
    {
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE
    },
    {
        mp_pose.PoseLandmark.LEFT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    }
]

ALLOWED_CONECTIONS_SIDE = [
    {
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE)
    },
    {
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE)
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
 Gera arquivo YOLO: <class> <x_center> <y_center> <w> <h>
Usando bounding box baseado nos landmarks.
"""
def yolo_arq(frame_num, img, dir_labels, dir_img, camera):
    # Seleciona de qual camera vai ser o risco detectado
    if camera == "front":
        hist = risk_detection_front
    elif camera == "side":
        hist = risk_detection_side
    else:
        return

    # Nome do arquivo
    arq_label = os.path.join(dir_labels, f"{frame_num:04d}.txt")
    arq_img = os.path.join(dir_img, f"{frame_num:04d}.jpg")

    risk_list = hist.get(frame_num, [])
    formated_list = "\n".join(risk_list)

    # Grava o arquivo texto no formato YOLO
    with open(arq_label, "w") as arq:
        arq.write(formated_list)

    # Grava o frame
    cv2.imwrite(arq_img, img)

"""
 Gera o label para cada risco detectado
"""
def yolo_label_txt(frame_num, center, bounding_box, risk_id, camera):
    # Seleciona de qual camera vai ser o risco detectado
    if camera == "front":
        hist = risk_detection_front
    elif camera == "side":
        hist = risk_detection_side
    else:
        return

    hist[frame_num].append(f"{risk_id} {center[0]:.6f} {center[1]:.6f} {bounding_box[0]:.6f} {bounding_box[1]:.6f}")

"""
    CÁLCULO DOS RISCOS:
"""

"""
• Distância dos pés:
 Cálculo da razão entre a distância dos ombros e a distância dos tornozelos
"""
def risk_stance_width(landmarks, frame_num):
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

    mid_shoulder = mid_point(left_shoulder, right_shoulder)
    mid_ankle = mid_point(left_ankle, right_ankle)
    center = mid_point(mid_shoulder, mid_ankle)

    boundingBox = (
        (max(left_shoulder[0], left_ankle[0], right_shoulder[0], right_ankle[0]) - min(right_shoulder[0], right_ankle[0], left_shoulder[0], left_ankle[0])),
        (max(right_ankle[1], left_ankle[1], right_shoulder[1], left_shoulder[1]) - min(right_shoulder[1], left_shoulder[1], right_ankle[1], left_ankle[1]))
    )

    if ratio < 0.8:
        if not RISK_DETECTED["stance_width"]:
            TEXT.append("Narrow Stance: " + f"{ratio:.2f}" + " at frame:" + str(frame_num))
            RISK_DETECTED["stance_width"] = True
        yolo_label_txt(frame_num, center, boundingBox, 0, "front")
    elif ratio > 1.2:
        if not RISK_DETECTED["stance_width"]:
            TEXT.append("Wide Stance: " + f"{ratio:.2f}" + ", at frame: " + str(frame_num))
            RISK_DETECTED["stance_width"] = True
        yolo_label_txt(frame_num, center, boundingBox, 1, "front")

"""
• Flexão Lateral do Tronco:
 Cálculo do angulo do vetor do tronco e um vetor perpendicular ao chão
"""
def risk_lateral_trunk(img, landmarks, frame_num):
    font = cv2.FONT_HERSHEY_SIMPLEX
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

    center = mid_point(mid_shoulder, mid_hip)

    # Referencia central do topo
    mid_top = np.array((mid_hip[0], 0))

    # Vetores usado como base de referencia e do quadril
    base_arr = create_vector(mid_top, mid_hip)
    trunk_arr = create_vector(mid_shoulder, mid_hip)

    angle = calculate_angle_arrays(base_arr, trunk_arr)

    boundingBox = (
        max(left_shoulder[0], left_hip[0], right_shoulder[0], right_hip[0]) - min(right_shoulder[0], right_hip[0], left_shoulder[0], left_hip[0]),
        max(right_hip[1], left_hip[1], right_shoulder[1], left_shoulder[1]) - min(right_shoulder[1], left_shoulder[1], right_hip[1], left_hip[1])
    )

    if angle > 7:
        if not RISK_DETECTED["lateral_trunk"]:
            RISK_DETECTED["lateral_trunk"] = True
            TEXT.append("Lateral Trunk detected at frame: " + str(frame_num))
        yolo_label_txt(frame_num, center, boundingBox, 2, "front")

    # Adiciona os vetores na imagem

    if points_selector == 2:
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
        left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        right_foot  = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

        center = mid_point(left_foot, right_foot)

        foot_assimetry = abs(left_foot[1] - right_foot[1])

        boundingBox = (
            (max(left_foot[0], right_foot[0], left_foot[0], right_foot[0]) - (min(left_foot[0], right_foot[0], left_foot[0], right_foot[0]))),
            (max(left_foot[1], right_foot[1], left_foot[1], right_foot[1]) - (min(left_foot[1], right_foot[1], left_foot[1], right_foot[1])))
        )

        if foot_assimetry > 0.05:
            if not RISK_DETECTED["foot_landing"]:
                RISK_DETECTED["foot_landing"] = True
                TEXT.append("Foot land assimetry at frame: " + str(frame_num))
            yolo_label_txt(frame_num, center, boundingBox, 3, "front")

"""
• Flexão de Joelho:
 Cálculo do angulo de flexão do joelho esquerdo
"""
def risk_knee_flexion(img, landmarks, frame_num):
    global knee_angle
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_color = (0, 255, 255)
    h, w, _ = img.shape

    # Coleta das referências do quadril, joelho e tornozelo
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

    # Criação dos vetores (Coxa e Canela)
    thigh_arr = create_vector(hip, knee)
    shin_arr = create_vector(ankle, knee)

    # Cálculo do angulo do joelho
    knee_angle = calculate_angle_arrays(shin_arr, thigh_arr)

    knee_px = (int(knee[0] * w), int(knee[1] * h))

    boundingBox = (
        (max(hip[0], knee[0], ankle[0]) - min(hip[0], knee[0], ankle[0])),
        (max(hip[1], knee[1], ankle[1]) - min(hip[1], knee[1], ankle[1])),
    )

    if points_selector == 0:
        cv2.putText(
            img,
            f"{knee_angle:.2f}",
            knee_px,
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

    if knee_angle < 30.0:
        if not RISK_DETECTED["knee_flexion"]:
            RISK_DETECTED["knee_flexion"] = True
            TEXT.append("Knee flexion at frame: " + str(frame_num))
        yolo_label_txt(frame_num, knee, boundingBox,4, "side")

"""
• Flexão de tronco
 Verifica se em algum momento do salto não houve a flexião da coluna.
"""
def risk_trunk_flexion(img, landmarks, frame_num):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_color = (0, 255, 255)
    h, w, _ = img.shape

    # Coleta das referencias de cada ponto (Ombro, quadril e joelho)
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

    # Criacção dos vetores (Tronco e Coxa)
    trunk_arr = create_vector(shoulder, hip)
    thigh_arr = create_vector(hip, knee)

    # Calculo do angulo
    trunk_angle = calculate_angle_arrays(trunk_arr, thigh_arr)

    hip_px = (int(hip[0] * w), int(hip[1] * h))

    boundingBox = (
        (max(hip[0], knee[0], shoulder[0]) - min(hip[0], knee[0], shoulder[0])),
        (max(hip[1], knee[1], shoulder[1]) - min(hip[1], knee[1], shoulder[1])),
    )

    if points_selector == 1:
        cv2.putText(
            img,
            f"{trunk_angle:.2f}",
            hip_px,
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

    if trunk_angle < 4.0:
        if not RISK_DETECTED["trunk_flexion"]:
            RISK_DETECTED["trunk_flexion"] = True
            TEXT.append("No trunk flexion at frame: " + str(frame_num))
        yolo_label_txt(frame_num, hip, boundingBox,5, "side")

"""
• Flexão plantar do tornozelo
 Cálculo da amplitude do tornozelo-pé.
"""
def risk_ankle_plantar_flexion(landmarks, frame_num):
    foot_index= [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

    center = mid_point(foot_index, ankle)

    dif = ankle[1] - foot_index[1]

    boundingBox = (
        (max(foot_index[0], ankle[0]) - min(foot_index[0], ankle[0])),
        (max(foot_index[1], ankle[1]) - min(foot_index[1], ankle[1]))
    )

    if dif > -0.020:
        if not RISK_DETECTED["ankle_plantar"]:
            RISK_DETECTED["ankle_plantar"] = True
            TEXT.append("Ankle Plantar Flexion at: " + str(frame_num))
        yolo_label_txt(frame_num, center, boundingBox, 6, "side")


def detect_risks(img_front, img_side, landmarks_front, landmarks_side, frame_num):
    risk_stance_width(landmarks_front, frame_num)
    risk_lateral_trunk(img_front, landmarks_front, frame_num)
    risk_feet_symmetry(landmarks_front, frame_num)
    risk_knee_flexion(img_side, landmarks_side, frame_num)
    risk_trunk_flexion(img_side, landmarks_side, frame_num)
    risk_ankle_plantar_flexion(landmarks_side, frame_num)

def main():
    global  JUMPING_TEXT
    global points_selector
    global risk_detection_front
    global risk_detection_side


    # Limpa e Cria as pastas para guardar as informacoes pra o treinamento do yolo
    dir_labels_front = "data_yolo/front/labels"
    dir_img_front = "data_yolo/front/images"

    dir_labels_side = "data_yolo/side/labels"
    dir_img_side = "data_yolo/side/images"


    if os.path.exists("data_yolo"):
        shutil.rmtree("data_yolo")

    os.makedirs(dir_labels_front)
    os.makedirs(dir_labels_side)
    os.makedirs(dir_img_front)
    os.makedirs(dir_img_side)

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

            risk_detection_front[frame_num] = []
            risk_detection_side[frame_num] = []

            ret_front, frame_front = cap_front.read()
            ret_side, frame_side = cap_side.read()

            frame_front_resized = cv2.resize(frame_front, res_yolo)
            frame_side_resized = cv2.resize(frame_side, res_yolo)

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

            # Armazena foto sem desenhos para o treinamento do yolo
            front_clear = image_bgr_front.copy()
            side_clear = image_bgr_side.copy()

            if not results_front.pose_landmarks or not results_side.pose_landmarks:
                continue

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
            draw_allowed_points(image_bgr_front, lm_front, ALLOWED_POINTS_FRONT[points_selector])
            draw_allowed_connections(image_bgr_front, lm_front, ALLOWED_CONECTIONS_FRONT[points_selector])

            draw_allowed_points(image_bgr_side, lm_side, ALLOWED_POINTS_SIDE[points_selector])
            draw_allowed_connections(image_bgr_side, lm_side, ALLOWED_CONECTIONS_SIDE[points_selector])

            detect_risks(image_bgr_front, image_bgr_side, lm_front, lm_side, frame_num)

            yolo_arq(frame_num, front_clear, dir_labels_front, dir_img_front, "front")
            yolo_arq(frame_num, side_clear, dir_labels_side, dir_img_side, "side")

            risk_detection_front.clear()
            risk_detection_side.clear()

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