# Recomendador de Playlists de Trilha Sonora Baseadas em Preferências de Jogos ou Animes.

Equipe: Alec de Jesus, Breno Passos, Felipe Brasileiro, Felipe Leão, Vinícius Souza

## Índice
- [Descrição do Projeto](#descrição-do-projeto)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação e Execução](#instalação-e-execução)
- [Ferramentas e Tecnologias](#ferramentas-e-tecnologias)
- [Conjunto de Dados](#conjunto-de-dados)

## Descrição do Projeto

Este projeto visa a criação de um sistema inteligente capaz de identificar obstáculos em tempo real na pista de veículos de alta performance, utilizando câmeras e algoritmos de visão computacional. O sistema será alimentado por um fluxo de vídeo, onde a inteligência artificial (IA) será responsável por detectar e classificar objetos no ambiente ao redor do veículo. Isso possibilita uma melhor tomada de decisão por parte do veículo autônomo ou assistido, ajudando na segurança e otimização da performance durante a condução.

## Tecnologias Utilizadas
- Python 3.x
- YoloV5
- OpenCV
- Bibliotecas: `torch`, `python display`, `cv2`
- Git/GitHub para controle de versão e colaboração

## Instalação e Execução

Para rodar o projeto localmente:

1. Clone o repositório:
    ```bash
    git clone https://github.com/VINI-DS001/street-object-detector
    ```

2. Instale as dependências:
    ```bash
    pip install torch
    ```

    ```bash
    pip install opencv-python
    ```

    ```bash
    pip install python display
    ```

## Ferramentas e tecnologias

Tecnologias Utilizadas

Front-End:
- ReactJS: Framework JavaScript moderno para construção da interface do usuário. A escolha de ReactJS permite a criação de uma interface rápida, reativa e escalável, com componentes reutilizáveis e fácil manutenção.
Inteligência Artificial:
- YOLO (You Only Look Once): YOLO é um dos modelos de detecção de objetos mais eficientes, utilizado para identificar e classificar objetos em imagens e vídeos em tempo real. Ele será utilizado para detectar os obstáculos na pista.
- OpenCV: Biblioteca de visão computacional open-source que será usada para processar os vídeos em tempo real, manipular frames e integrar as saídas do modelo YOLO ao fluxo de vídeo.

## Conjunto de Dados

O conjunto de dados original utilizado para o projeto inclui aproximadamente 22 mil imagens de pessoas, veículos entre outros obstáculos na estrada utilizados para o treinamento do detector.

 - [Dataset: Detecção de objetos](https://www.kaggle.com/datasets/rezafazel63/street-object-detection-dataset)