import cv2
import numpy as np

from .colors import GOLD
from .mask import Line, Shape


def frame_to_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    background = np.zeros(frame.shape, dtype=np.uint8)
    
    sensitivity = 20

    # Range de cores - limite inferior
    lower_range = np.array([110 - sensitivity, 50, 50])

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
    return np.where(mask_3_channels == 0, background, frame)


def hsv_edges(hsv, threshold_1=80, threshold_2=200):
    return cv2.Canny(hsv, threshold_1, threshold_2)


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


        

