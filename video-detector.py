import warnings

import cv2
import torch
from PIL import Image

# Função para detectar objetos em um único frame
def detect_objects_in_frame(model, frame):
    # Converte o frame de OpenCV (BGR) para PIL (RGB)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Realiza a detecção de objetos no frame usando o modelo YOLOv5
    results = model(img)

    # Renderiza as detecções na imagem (adiciona as caixas de detecção)
    results.render()  # Renderiza as caixas de detecção diretamente

    # Retorna o frame com as caixas de detecção desenhadas
    return results.ims[0]  # Acessando o frame processado

# Função para processar o vídeo
def detect_objects_in_video(video_path, model):
    # Abra o arquivo de vídeo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    while True:
        # Captura frame a frame
        ret, frame = cap.read()

        if not ret:
            break  # Se não houver mais frames, saia do loop

        # Realiza a detecção no frame
        processed_frame = detect_objects_in_frame(model, frame)

        # Exibe o frame com as detecções no OpenCV
        cv2.imshow('Detecção de Objetos no Vídeo', processed_frame)

        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libere a captura de vídeo e feche as janelas
    cap.release()
    cv2.destroyAllWindows()

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Caminho para o vídeo gravado
video_path = 'videos/Tesla-Self-Driving.mp4'  # Substitua pelo caminho do seu vídeo

# Realizar a detecção no vídeo
detect_objects_in_video(video_path, model)

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
