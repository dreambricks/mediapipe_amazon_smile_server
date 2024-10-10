from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from utils import FaceLandmarks
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignorar alguns warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Inicializar Flask e SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')  # Usando 'threading' como modo assíncrono

draw_stats = True
draw_mask = False

fl = FaceLandmarks(static_image_mode=False)
emotions = ['happy', 'laughing', 'neutral']
emotions_pt = ['sorrindo', 'rindo', 'neutro']
emotions_idx_sorted = [1, 2, 0]

# Carregar o modelo
with open('./model4.pkl', 'rb') as f:
    model = pickle.load(f)

def process_frame(frame):
    """Processar o quadro de vídeo para detecção de emoções."""
    try:
        face_landmarks = fl.get_face_landmarks(frame, draw=draw_mask)
        if len(face_landmarks) == 1404:
            output = model.predict_proba([face_landmarks])
            if draw_stats:
                max_val = max(output[0])
                for idx, e_idx in enumerate(emotions_idx_sorted):
                    e = emotions_pt[idx]
                    text = f"{e} : {output[0][idx] * 100:.0f}%"
                    color = (0, 255, 0) if output[0][idx] == max_val else (0, 0, 255)
                    cv2.putText(frame, text, (10, frame.shape[0] - 10 - (len(emotions) - e_idx - 1) * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    except Exception as e:
        print(f"Error processing frame: {e}")
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Lidar com quadros de vídeo recebidos e enviar quadros processados de volta."""
    frame_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(frame_data, np.uint8)

    if np_arr.size == 0:
        return
    try:
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except cv2.error as e:
        return
    processed_frame = process_frame(frame)
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_data = base64.b64encode(buffer).decode('utf-8')
    emit('processed_frame', f"data:image/jpeg;base64,{processed_frame_data}")

if __name__ == '__main__':
    # Iniciar o servidor Flask com SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
