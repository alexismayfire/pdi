import cv2
import numpy as np

from .colors import GOLD
from .mask import Line, Shape


def frame_to_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    background = np.zeros(frame.shape, dtype=np.uint8)
    
    sensitivity = 30

    # Range de cores - limite inferior
    lower_range = np.array([110 - sensitivity, 70, 50])

    # Range do azul - limite superior
    upper_range = np.array([130 + sensitivity, 255, 255])

    # O inRange cria uma máscara, setando pixels que estão
    # entre lower_range e upper_range, ou seja, apenas os 
    # pixels azuis de interesse
    mask = cv2.inRange(hsv, lower_range, upper_range)
    # Como vamos combinar a máscara com o frame, precisa ter 3 canais
    mask_3_channels = np.zeros(frame.shape, dtype=np.uint8)
    for ch in range(2):
        mask_3_channels[:, :, ch] = mask[:, :]

    # Agora, com o np.where retornamos o background onde a 
    # máscara não está setada, e o frame original caso esteja
    masked_hsv = np.where(mask_3_channels == 0, background, hsv)
    return masked_hsv


def hsv_edges(hsv, threshold_1=80, threshold_2=200):
    width, height, _ = hsv.shape

    # Aqui, np.where vai retornar uma matriz com 3 'canais',
    # setando [cmp, cmp, cmp] de acordo com o resultado da comparação
    # Ou seja, no ponto hsv[a, b, c]:
    #   - O valor de H >= 10
    #   - O valor de S >= 10
    #   - O valor de V >= 0
    # Então, binarized[a, b, c] = [255, 255, 255]
    # Agora, caso hsv[a, b, c] seja:
    #   - O valor de H < 10
    #   - O valor de S < 10
    #   - O valor de V >= 0
    # Então, binarized[a, b, c] = [0, 0, 255]
    # Como estamos mais interessados no Hue, então usamos ele abaixo no retorno
    # Lembrando que o parâmetro 'hsv' aqui já vem filtrado previamente, 
    # com frequências de azul, e é por isso que podemos usar [10, 10, 0]
    # OBS: por alguma razão, o V sempre vem 0 da frame_to_hsv()...
    binarized = np.where(hsv >= np.array([10, 10, 0]), 255, 0)

    # Como Hough aceita imagens 8UC1, ou seja, 8bpp, unsigned e 1 canal,
    # pegamos apenas o primerio canal e usa o cast para np.uint8
    return binarized[:, :, 0].astype(np.uint8)


def draw_line(image, line, color, thickness=5):
    cv2.line(image, line.start, line.end, color, thickness)


def draw_lines(image, lines, color, thickness=5):
    for line in lines:
        draw_line(image, line, color, thickness)


def tattoo_image(size):
    tattoo = cv2.imread("tattoo1.png")
    tattoo = cv2.resize(tattoo, size)

    alpha = tattoo[:, :, 0] / 255

    return alpha

def draw_tattoo(frame, shape: Shape):
    if shape.left and shape.right and shape.bottom:
        # lines = [shape.left, shape.right, shape.bottom]
        # draw_lines(frame, lines, GOLD)

        frame_y_range = range(Shape.y_start(), Shape.y_end())
        frame_x_range = range(Shape.left.x, Shape.right.x)

        tattoo = tattoo_image(shape.size())
        tattoo_width, tattoo_height = tattoo.shape
        tattoo_y_range = range(tattoo_width)
        tattoo_x_range = range(tattoo_height)

        for y, y2 in zip(frame_y_range, tattoo_y_range):
            for x, x2 in zip(frame_x_range, tattoo_x_range):
                if tattoo[y2, x2] == 0:
                    frame[y, x] = np.array([
                        tattoo[y2, x2], tattoo[y2, x2], tattoo[y2, x2]
                    ])


        

