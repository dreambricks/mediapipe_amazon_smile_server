import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import mediapipe as mp
import os

# Classe FaceLandmarks conforme fornecido
class FaceLandmarks:
    def __init__(self, static_image_mode=True):
        self.static_image_mode = static_image_mode
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                            max_num_faces=1,
                                            min_detection_confidence=0.5)

    def get_face_landmarks(self, image, draw=False):
        image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_input_rgb)
        image_landmarks = []

        if results.multi_face_landmarks:
            if draw:
                mp_drawing = mp.solutions.drawing_utils
                drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

            ls_single_face = results.multi_face_landmarks[0].landmark
            xs_ = []
            ys_ = []
            zs_ = []
            for idx in ls_single_face:
                xs_.append(idx.x)
                ys_.append(idx.y)
                zs_.append(idx.z)

            sx = max(xs_) - min(xs_)
            sy = max(ys_) - min(ys_)
            sz = max(zs_) - min(zs_)
            s = max([sx, sy, sz])
            if s == 0.0: s = 1.0

            for j in range(len(xs_)):
                image_landmarks.append((xs_[j] - min(xs_)) / s)
                image_landmarks.append((ys_[j] - min(ys_)) / s)
                image_landmarks.append((zs_[j] - min(zs_)) / s)

        return image_landmarks

# Função para coletar dados
def collect_data(image_folder, labels):
    landmarks_list = []
    labels_list = []

    fl = FaceLandmarks()

    for label in labels:
        # Caminho das imagens
        folder_path = os.path.join(image_folder, label)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Carregar a imagem
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Extrair landmarks
            landmarks = fl.get_face_landmarks(image)
            if landmarks:
                landmarks_list.append(landmarks)
                labels_list.append(label)

    return np.array(landmarks_list), np.array(labels_list)

# Configurações
image_folder = 'captured_images'
labels = ['happy', 'laughing', 'neutral']

# Coletar dados
X, y = collect_data(image_folder, labels)

# Codificar labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Treinar o modelo
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Avaliar o modelo
accuracy = model.score(X_test, y_test)
print(f'Acurácia: {accuracy:.2f}')

# Salvar o modelo
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Salvar o encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Modelo e encoder salvos com sucesso!")
