from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import mediapipe as mp

# Inicializar Flask e SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Inicializar MediaPipe para detecção de mãos, por exemplo
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


# Função para processar a imagem
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def process_frame(frame):
    # Converter a imagem para escala de cinza, necessário para os classificadores Haar
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos no frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Para cada rosto detectado, desenhar um retângulo e procurar por sorrisos
    for (x, y, w, h) in faces:
        # Desenhar um retângulo ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Região do rosto em cinza para detectar sorrisos
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detectar sorrisos na região do rosto
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

        # Para cada sorriso detectado, desenhar um retângulo
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.putText(frame,"Smiling",(x,y-50),cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0),3,cv2.LINE_AA)

    return frame


# Rota para a página HTML
@app.route('/')
def index():
    return render_template('index.html')


# WebSocket para receber o stream de vídeo do cliente
@socketio.on('video_frame')
def handle_video_frame(data):
    # Decodificar o frame recebido do cliente
    frame_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(frame_data, np.uint8)

    if np_arr.size == 0:
        return

    try:
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except cv2.error as e:
        return

    # Processar o frame com MediaPipe
    processed_frame = process_frame(frame)

    # Codificar o frame processado para enviar de volta ao cliente
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_data = base64.b64encode(buffer).decode('utf-8')

    # Enviar o frame processado de volta ao cliente
    emit('processed_frame', f"data:image/jpeg;base64,{processed_frame_data}")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
