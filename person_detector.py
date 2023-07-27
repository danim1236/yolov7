import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xywh2xyxy

class PersonDetector:
    def __init__(self, weights_path='./yolov7.pt', device='cpu', img_size=640, conf_thres=0.25, iou_thres=0.45, min_confidence=0.6):
        # Carregar o modelo YOLOv7
        self.device = device
        print(f"Using device: {device}")
        self.model = attempt_load(weights_path, map_location=device)
        self.model.to(device).eval()

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = self.model.names if hasattr(self.model, 'names') else self.model.module.names
        self.min_confidence = min_confidence

    def process(self, frame):
        # Obter o tamanho original da imagem
        h_orig, w_orig = frame.shape[:2]

        # Redimensionar o frame para o tamanho esperado pelo modelo
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)[0]

        # Filtro de detecção usando NMS (Non-Maximum Suppression)
        detections = non_max_suppression(output, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

        # Processar detecções
        person_detections = []
        for detection in detections:
            if detection is not None and len(detection) > 0:
                detection[:, :4] = scale_coords(img_tensor.shape[2:], detection[:, :4], frame.shape).round()

                # Calcular os fatores de escala horizontal e vertical
                scale_x = w_orig / self.img_size
                scale_y = h_orig / self.img_size

                for *xyxy, conf, cls in detection:
                    if self.names[int(cls)] == 'person' and conf >= self.min_confidence:  # Verificar se é uma pessoa e confiança mínima
                        # Ajustar as coordenadas de volta à proporção original
                        x1, y1, x2, y2 = xyxy
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y

                        bbox = [x1.item(), y1.item(), x2.item(), y2.item()]
                        person_detections.append(bbox)

        return person_detections
