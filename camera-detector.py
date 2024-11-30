import cv2
from ultralytics import YOLO

# Função para detectar objetos em um único frame
def detect_objects_in_frame(model, frame):
    # Obtemos as dimensões do frame
    img_height, img_width, _ = frame.shape

    # Realiza a detecção de objetos no frame, limitando às classes especificadas
    results = model.predict(source=frame, classes=[0, 2], save=False, conf=0.5)

    # Processar as detecções
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])  # Classe do objeto
            conf = box.conf[0]  # Confiança da detecção
            xmin, ymin, xmax, ymax = map(int, xyxy)

            # Determinar a ação com base na classe
            if cls == 2:  # Classe "car"
                print(f"Objeto detectado: Carro (Confiança: {conf:.2f}) -> Ação: Reduza a velocidade ou Freie.")
            elif cls == 0:  # Classe "person":
                # Calcular o centro do bounding box
                center_x = (xmin + xmax) / 2

                # Determinar se está na metade esquerda ou direita
                if center_x < img_width / 2:  # Metade esquerda
                    print(f"Objeto detectado: Pessoa (Confiança: {conf:.2f}) na esquerda -> Ação: Desviar para a direita.")
                else:  # Metade direita
                    print(f"Objeto detectado: Pessoa (Confiança: {conf:.2f}) na direita -> Ação: Desviar para a esquerda.")

    return frame  # Retornamos o frame original para exibição

# Função para processar vídeo ao vivo (câmera)
def detect_objects_in_live_video(model):
    # Abrir a câmera (ID padrão 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    print("Pressione 'q' para sair.")

    while True:
        # Captura frame a frame
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar o frame da câmera.")
            break

        # Realiza a detecção no frame
        _ = detect_objects_in_frame(model, frame)

        # Exibe o frame sem renderização de bounding boxes (somente log no terminal)
        cv2.imshow('Detecção ao Vivo', frame)

        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libere a captura de vídeo e feche as janelas
    cap.release()
    cv2.destroyAllWindows()

# Carregar o modelo YOLOv8
model = YOLO('yolov8n.pt')

# Realizar a detecção ao vivo
detect_objects_in_live_video(model)
