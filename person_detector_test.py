import cv2
from person_detector import PersonDetector

if __name__ == '__main__':
    video_path = './video_test_1.mp4'
    detector = PersonDetector()

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processar o frame para obter as detecções de pessoas
        person_detections = detector.process(frame)

        print(person_detections)

        # Desenhar caixas delimitadoras nas detecções de pessoas
        for bbox in person_detections:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Exibir o frame com as detecções
        cv2.imshow('Detection', frame)

        # Aguardar 1 milissegundo e verificar se uma tecla foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar o objeto VideoCapture e fechar a janela
    cap.release()
    cv2.destroyAllWindows()
