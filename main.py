import cv2
import numpy as np
import pytesseract
import pyautogui
import time
from googletrans import Translator
import re  # Importa o módulo para expressões regulares

# Defina o caminho para o executável Tesseract se necessário
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Crie um objeto Translator
translator = Translator()

# Configuração da detecção de texto
custom_config = r'--oem 3 --psm 6'

# Defina o tamanho da tela
screen_size = pyautogui.size()
screen_width, screen_height = screen_size

# Defina o tamanho da região de interesse (ROI) e ajuste conforme necessário
roi_width, roi_height = 1200, 230  # Ajuste conforme necessário
x = (screen_width - roi_width) // 2
y = (screen_height - roi_height) // 2  # Posição original da ROI

# Deslocamento para mover a ROI para baixo
deslocamento_y = 370  # Ajuste o valor conforme necessário
y += deslocamento_y

# Defina o codec e crie um objeto VideoWriter para gravar somente a ROI
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (roi_width, roi_height))

# Variável para armazenar o texto da última iteração
last_detected_text = ""
last_detection_time = time.time()
detection_interval = 2  # Intervalo mínimo entre detecções (em segundos)
min_length = 3  # Comprimento mínimo do texto para consideração
similarity_threshold = 0.8  # 80% de similaridade

def string_similarity(str1, str2):
    """ Calcula a similaridade entre duas strings """
    if not str1 or not str2:
        return 0
    set1, set2 = set(str1), set(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

while True:
    # Capture a tela somente na região de interesse (ROI)
    img = pyautogui.screenshot(region=(x, y, roi_width, roi_height))
    
    # Converta a imagem para um array numpy
    frame = np.array(img)
    
    # Converta de RGB para BGR (OpenCV usa BGR por padrão)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Verifique se a imagem foi capturada corretamente
    if frame is None or frame.size == 0:
        print("Erro: Imagem capturada está vazia.")
        continue

    # Converta a ROI para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplica binarização adaptativa
    _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Aplica dilatação e erosão para melhorar a detecção de texto
    kernel = np.ones((3, 3), np.uint8)
    binary_frame = cv2.dilate(binary_frame, kernel, iterations=1)
    binary_frame = cv2.erode(binary_frame, kernel, iterations=1)
    
    # Detecta texto na ROI
    detected_text = pytesseract.image_to_string(binary_frame, config=custom_config).strip()
    
    # Filtra o texto para manter apenas letras, números, vírgulas, pontos e pontos de interrogação
    filtered_text = ' '.join(re.findall(r'[A-Za-zÀ-ÿ0-9,.\?!]+', detected_text))
    
    # Verifica se o texto filtrado é novo e relevante
    if filtered_text and len(filtered_text) >= min_length:
        current_time = time.time()
        if (current_time - last_detection_time) > detection_interval:
            if string_similarity(filtered_text, last_detected_text) < similarity_threshold:
                # Traduz o texto detectado
                translation = translator.translate(filtered_text, src='pt', dest='en')
                translated_text = translation.text
                
                # Imprime o texto detectado e traduzido
                print(f"Texto Detectado: {filtered_text}")
                print(f"Texto Traduzido: {translated_text}")
                
                # Atualiza o texto da última detecção
                last_detected_text = filtered_text
                last_detection_time = current_time
    
    # Adicione um retângulo vermelho ao redor da região de interesse
    cv2.rectangle(frame, (0, 0), (roi_width, roi_height), (0, 0, 255), 2)
    
    # Escreva o frame no arquivo de vídeo
    out.write(frame)
    
    # Mostre o frame em uma janela (opcional)
    cv2.imshow("Recording", frame)
    
    # Interrompa a gravação quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os objetos VideoWriter e feche todas as janelas
out.release()
cv2.destroyAllWindows()
