import cv2
import numpy as np
import time

from .colors import BLUE, GOLD, RED
from .detector import hough_detection, match_line_with_shape
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

        lines = hough_detection(frame, edges)
        for line in lines:
            # Na classe Shape é que vamos setar o que foi encontrado!
            # Como ela só tem atributos de classe, não estamos instanciando um objeto,
            # então não vai "perder" os valores entre um frame e outro
            match_line_with_shape(frame, line, coords, Shape)

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
