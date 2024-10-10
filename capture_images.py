import cv2
import os


def capture_images(output_folder, labels, num_images=10):
    # Cria as pastas para cada label se não existirem
    for label in labels:
        label_folder = os.path.join(output_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

    # Inicializa a captura de vídeo da câmera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    print("Pressione a tecla correspondente à label para capturar as imagens:")
    print("1 - happy")
    print("2 - laughing")
    print("3 - neutral")
    print("Pressione 'q' para sair.")

    count = 0
    current_label = None

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame.")
            break

        # Exibe o frame na tela
        cv2.imshow('Captura de Imagem', frame)

        # Aguarda uma tecla
        key = cv2.waitKey(1)

        # Verifica a tecla pressionada para selecionar a label
        if key == ord('1'):
            current_label = 'happy'
            print("Label selecionada: happy")
        elif key == ord('2'):
            current_label = 'laughing'
            print("Label selecionada: laughing")
        elif key == ord('3'):
            current_label = 'neutral'
            print("Label selecionada: neutral")
        elif key == ord('s') and current_label:
            image_path = os.path.join(output_folder, current_label, f'image_{count}.jpg')
            cv2.imwrite(image_path, frame)
            print(f'Imagem salva em: {image_path}')
            count += 1
        elif key == ord('q'):
            break

    # Libera a captura e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()


# Configurações
output_folder = 'captured_images5'  # Pasta onde as imagens serão salvas
labels = ['happy', 'laughing', 'neutral']  # Labels para as pastas
num_images = 30  # Número de imagens a serem capturadas

# Chama a função de captura de imagens
capture_images(output_folder, labels, num_images)
