import cv2
import numpy as np
import time

from .colors import BLUE, GOLD, RED
from .detector import hough_detection, match_line_with_shape, scale_detection
from .mask import Shape, get_mask_coordinates
from .image import draw_lines, draw_tattoo, frame_to_hsv, hsv_edges


def run():
    cap = cv2.VideoCapture(0)

    # Confere se a webcam pode ser acessada
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Booleano simples para a primeira iteração
    set_mask_coords = True
    # Esse valor vai ser setado somente na primeira iteração do programa
    # Vai servir para retornar a posição da máscara caso a detecção tenha sumido
    # Por exemplo, moveu muito rápido. Não pode ser alterado!
    mask_coords = None
    # Deve guardar sempre a última coordenada identificada da shape, ou seja,
    # deve ser um vetor com 3 Line, na forma: [LEFT_LINE, RIGHT_LINE, BOTTOM_LINE]
    coords = None
    TIME_COUNTER = 1

    # A cada frame processado
    while True:
        ret, frame = cap.read()

        # Giramos o frame, porque ele vem espelhado na captura
        frame = cv2.flip(frame, 1)
            
        hsv = frame_to_hsv(frame)
        edges = hsv_edges(hsv)

        # Na primeira iteração, esse valor é verdadeiro
        # Precisamos disso apenas porque somente dentro do laço
        # vamos ter acesso ao frame, para executar a get_mask_coordinates
        # Ou seja, SÓ RODA NA PRIMEIRA ITERAÇÃO!
        if set_mask_coords:
            mask_coords = get_mask_coordinates(frame)
            coords = get_mask_coordinates(frame)
            set_mask_coords = False

        if Shape.detected() is False:
            lines = hough_detection(frame, edges)
            for line in lines:
                # Na classe Shape é que vamos setar o que foi encontrado!
                # Como ela só tem atributos de classe, não estamos instanciando um objeto,
                # então não vai "perder" os valores entre um frame e outro
                match_line_with_shape(frame, line, coords, Shape)
        else:
            y_start = Shape.y_start()
            y_end = Shape.y_end()
            x_start = Shape.left.x
            x_end = Shape.right.x
            template = frame[y_start:y_end, x_start:x_end]
            width_t, height_t, ch = template.shape

            dif = np.zeros(template.shape)
            width, height, ch = frame.shape
            last_y = 0
            last_x = 0            

            while last_y < height:
                for y, y2 in zip(range(last_y, height), range(height_t)):
                    while last_x < width:
                        for x, x2 in zip(range(last_x, width), range(width_t)):
                            for ch in range(ch):
                                # Computaria diferenças???
                                dif[y, x, ch] = abs(frame[y2, x2, ch] - template[y2, x2, ch])
                                
                        last_x = last_x + 10

                last_y = last_y + 10


        TIME_COUNTER = TIME_COUNTER + 1
        if TIME_COUNTER % 20 == 0:
            # TODO: como resetar coords??
            if Shape.detected():
                if TIME_COUNTER % 600 == 0:
                    print(f'RESETOU AS COORDENADAS INICIAIS | TIMER: {TIME_COUNTER}')
                    coords = mask_coords
                else:
                    coords = [Shape.left, Shape.right, Shape.bottom]
            Shape.left = None
            Shape.right = None
            Shape.bottom = None

        # Desenha o shape inicial na imagem capturada
        if Shape.detected():
            draw_tattoo(frame, Shape)
            coords = [Shape.left, Shape.right, Shape.bottom]
        else:
            draw_lines(frame, mask_coords, BLUE)

        height, width, channels = frame.shape
        side = np.zeros((height, width * 2, channels), dtype=np.uint8)
        side[:, :width, :] = frame[:, :, :]
        # side[:, width:, :] = hsv[:, :, :]
        side[:, width:, 0] = edges[:, :]
        side[:, width:, 1] = edges[:, :]
        side[:, width:, 2] = edges[:, :]
        cv2.imshow("Input", side)
        c = cv2.waitKey(1)

        # Valor ASCII do Esc
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
