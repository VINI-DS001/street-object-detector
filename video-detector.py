import warnings
import cv2
from ultralytics import YOLO
from PIL import Image

# Função para detectar objetos em um único frame
def detect_objects_in_frame(model, frame):
    # Obtemos as dimensões do frame
    img_height, img_width, _ = frame.shape

    # Converte o frame de OpenCV (BGR) para PIL (RGB)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Realiza a detecção de objetos no frame, limitando às classes especificadas
    results = model.predict(source=img, classes=[0, 2], save=False, conf=0.5)

    # Processar as detecções
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])  # Classe do objeto
            conf = box.conf[0]  # Confiança da detecção
            xmin, ymin, xmax, ymax = map(int, xyxy)

            # Calcular o centro do bounding box
            center_x = (xmin + xmax) / 2

            # Determinar a ação com base na classe
            if cls == 2:  # Classe "car"
                print(f"Objeto detectado: Carro (Confiança: {conf:.2f}) -> Ação: Reduza a velocidade ou Freie.")
            elif cls == 0:  # Classe "person"
                if center_x < img_width / 2:  # Metade esquerda
                    print(f"Objeto detectado: Pessoa (Confiança: {conf:.2f}) na esquerda -> Ação: Desviar para a direita.")
                else:  # Metade direita
                    print(f"Objeto detectado: Pessoa (Confiança: {conf:.2f}) na direita -> Ação: Desviar para a esquerda.")

            # Determinar cor e texto baseado na classe
            if cls == 0:  # Classe "person"
                color = (0, 0, 255)  # Vermelho
                label = f"Person {conf:.2f}"
            elif cls == 2:  # Classe "car"
                color = (255, 0, 0)  # Azul
                label = f"Car {conf:.2f}"
            else:
                continue

            # Desenhar bounding box no frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Função para processar o vídeo
def detect_objects_in_video(video_path, model):
    # Abra o arquivo de vídeo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    print("Processando o vídeo. Pressione 'q' para sair.")

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

# Carregar o modelo YOLOv8
model = YOLO('yolov8n.pt')

# Caminho para o vídeo gravado
video_path = 'videos/Tesla-Self-Driving.mp4'  # Substitua pelo caminho do seu vídeo

# Realizar a detecção no vídeo
detect_objects_in_video(video_path, model)

# Ignorar avisos do PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
