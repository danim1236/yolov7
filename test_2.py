import cv2
import time
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xywh2xyxy

def detect_person_in_video(video_path, weights_path, img_size=640, conf_thres=0.25, iou_thres=0.45):
    # Carregar o modelo YOLOv7
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = attempt_load(weights_path, map_location=device)
    model.to(device).eval()
    
    # Carregar nomes das classes
    names = model.names if hasattr(model, 'names') else model.module.names

    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo!")
        return

    # Configurar a janela para exibir o vídeo
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 800, 600)

    frame_count = 0
    start_time = time.time()

    while True:
        # Ler um frame do vídeo
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar o frame para o tamanho esperado pelo modelo
        frame = cv2.resize(frame, (img_size, img_size))
        img_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(img_tensor)[0]

        # Filtro de detecção usando NMS (Non-Maximum Suppression)
        detections = non_max_suppression(output, conf_thres=conf_thres, iou_thres=iou_thres)

        # Processar detecções
        for detection in detections:
            if detection is not None and len(detection) > 0:
                detection[:, :4] = scale_coords(img_tensor.shape[2:], detection[:, :4], frame.shape).round()

                for *xyxy, conf, cls in detection:
                    if names[int(cls)] == 'person':  # Verificar se é uma pessoa
                        # Desenhar caixa delimitadora azul e adicionar label 'person'
                        frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                        frame = cv2.putText(frame, names[int(cls)], (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Calcular e exibir os FPS em tempo real
        frame_count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Exibir o frame com as detecções e os FPS
        cv2.imshow('Detection', frame)

        # Aguardar 1 milissegundo e verificar se uma tecla foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar o objeto VideoCapture e fechar a janela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = './video_test_1.mp4'
    weights_path = './yolov7.pt'
    detect_person_in_video(video_path, weights_path)
