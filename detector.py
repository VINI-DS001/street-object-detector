# run_object_detection.py

import os
import random
from PIL import Image
from IPython.display import display
import torch

# Função para listar as extensões de arquivos em um diretório
def get_file_extensions(folder_path):
    files = os.listdir(folder_path)
    file_extensions = {os.path.splitext(file)[1] for file in files}
    return file_extensions

# Função para detectar objetos nas imagens
def detect_objects(model, image_paths):
    # Realiza a detecção de objetos nas imagens
    results = model(image_paths)
    
    # Print e visualiza os resultados da detecção
    results.print()
    results.show()
    
    return results

# Caminho base para as imagens
base_path = r'images\images\images'

# Listar todas as imagens no diretório
all_images = os.listdir(base_path)

# Escolher uma imagem aleatória
random_image = random.choice(all_images)

# Abrir e exibir a imagem escolhida
img = Image.open(os.path.join(base_path, random_image))
display(img)

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Usar a imagem escolhida para inferência
imgs = img  # batch de imagens

# Realizar a inferência com o modelo YOLOv5
results = model(imgs)

# Realizar detecção de objetos nas imagens passadas como argumento
detection_results = detect_objects(model, imgs)

# Lista de caminhos de imagens para mais detecções
image_paths = [
    f"images/images/test/test1.jpg",
    f"images/images/test/test2.jpg",
    f"images/images/test/test3.jpg",
    f"images/images/test/test5.jpg",
    #f"{base_path}/1478020515199458307.jpg",
    #f"{base_path}/1478020231691535596.jpg",
    #f"{base_path}/1478020351195471769.jpg",
    #f"{base_path}/1478898499983147215.jpg",
    #f"{base_path}/1478898651375864863.jpg",
    #f"{base_path}/1479506165491761103.jpg",
    #f"{base_path}/1478898957016224931.jpg",
    #f"{base_path}/th1.jpg",
    #f"{base_path}/GRdCC.jpg"
]

# Realizar a detecção de objetos para as imagens listadas
results = detect_objects(model, image_paths)

# Extrair os resultados da detecção como um DataFrame Pandas
data_frame = results.pandas().xyxy[0]

# Exibir os resultados da detecção
print("Resultados da Detecção de Objetos:")
print(data_frame)
