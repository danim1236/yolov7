import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xywh2xyxy

def detect_person_in_image(image_path, weights_path, img_size=640, conf_thres=0.25, iou_thres=0.45):
    # Carregar o modelo YOLOv7
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = attempt_load(weights_path, map_location=device)
    model.to(device).eval()
    
    # Carregar nomes das classes
    names = model.names if hasattr(model, 'names') else model.module.names

    # Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        print("Erro: Imagem não encontrada!")
        return
 
    # Redimensionar a imagem para o tamanho esperado pelo modelo
    img = cv2.resize(img, (img_size, img_size))
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)[0]

    # Filtro de detecção usando NMS (Non-Maximum Suppression)
    detections = non_max_suppression(output, conf_thres=conf_thres, iou_thres=iou_thres)

    # Processar detecções
    for detection in detections:
        if detection is not None and len(detection) > 0:
            detection[:, :4] = scale_coords(img_tensor.shape[2:], detection[:, :4], img.shape).round()

            for *xyxy, conf, cls in detection:
                if names[int(cls)] == 'person':  # Verificar se é uma pessoa
                    # Desenhar caixa delimitadora azul e adicionar label 'person'
                    img = cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    img = cv2.putText(img, names[int(cls)], (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Exibir a imagem com as detecções
    cv2.imshow('Detection', img)

    # Esperar até que uma tecla seja pressionada e fechar a janela
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = './people_walking_around_city.jpeg'
    weights_path = './yolov7.pt'
    detect_person_in_image(image_path, weights_path)
