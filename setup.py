# setup_environment.py

# Instalar pacotes necessários
import subprocess
import sys

# Função para instalar pacotes
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Instalar as dependências
def install_dependencies():
    # Pacotes principais
    install_package("torch")
    install_package("Pillow")
    install_package("IPython")
    
    # Caso precise instalar o YOLOv8 (se não já estiver instalado)
    install_package("git+https://github.com/ultralytics/ultralytics.git")

# Rodar a instalação
if __name__ == "__main__":
    install_dependencies()
