# run_object_detection_yolov8.py

import os
from PIL import Image
from IPython.display import display
from ultralytics import YOLO

# Função para processar as detecções e decidir ações
def process_detections(results, img):
    # Obtemos as dimensões da imagem
    img_width, img_height = img.size

    # Iterar sobre cada detecção
    for result in results:
        for box in result.boxes:  # Cada box é uma detecção individual
            cls = int(box.cls[0])  # Classe do objeto
            conf = box.conf[0]  # Confiança da detecção
            xyxy = box.xyxy[0].cpu().numpy()  # Coordenadas do bounding box
            xmin, ymin, xmax, ymax = xyxy

            # Determinar a ação com base na classe
            if cls == 2:  # Classe "car"
                print(f"Objeto detectado: Carro (Confiança: {conf:.2f}) -> Ação: Reduza a velocidade ou Freie.")
            elif cls == 0:  # Classe "person"
                # Calcular o centro do bounding box
                center_x = (xmin + xmax) / 2

                # Determinar se está na metade esquerda ou direita
                if center_x < img_width / 2:  # Metade esquerda
                    print(f"Objeto detectado: Pessoa (Confiança: {conf:.2f}) na esquerda -> Ação: Desviar para a direita.")
                else:  # Metade direita
                    print(f"Objeto detectado: Pessoa (Confiança: {conf:.2f}) na direita -> Ação: Desviar para a esquerda.")

# Caminho para a imagem de teste
test_image_path = "images/images/test/test1.jpg"

# Exibir a imagem original
img = Image.open(test_image_path)
display(img)

# Carregar o modelo YOLOv8
model = YOLO('yolov8n.pt')

# Realizar inferência na imagem de teste, limitando às classes de interesse (0: "person", 2: "car")
results = model.predict(source=test_image_path, classes=[0, 2])

# Processar os resultados e decidir ações
process_detections(results, img)

# Exibir a imagem com os resultados da detecção
# results[0].plot() desenha os bounding boxes na imagem original e retorna a imagem plotada
annotated_image = results[0].plot()  # Gera a imagem anotada
display(Image.fromarray(annotated_image))  # Exibe a imagem anotada
