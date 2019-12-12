import cv2
import math
import numpy as np
from scipy.stats import linregress


RED = (0, 0, 255)
WHITE = (255, 255, 255)


def get_perpendicular_line(x, y, slope, image_width, desired_line_size):
    """
    Essa funçao retorna uma reta que é perpendicular à reta com 
    inclinação = slope, e que contém os pontos (x, y)
    """

    # Para achar uma reta na forma y = mx + b, perpendicular à 
    # reta que passa pelo ponto (x, y)
    perpendicular_slope = np.reciprocal(slope)
    intercept = y + (perpendicular_slope * x)
    perpendicular_slope *= -1

    # Nosso target é gerar uma reta com o mesmo comprimento que a bottom_line
    closest_distance = 999
    point = None

    # Dependendo da inclinação da reta que queremos buscar a perpendicular,
    # podemos ir para a direita ou para a esquerda
    if slope < 0:
        # Talvez x seja grande, então vamos decrescendo até zero
        # Dessa forma, com a cláusula de parada no loop, nunca vamos de fato
        # até zero. Se fosse um range convencional (0, x) em um x grande,
        # o for acabaria iterando muito mais do que o necessário
        test_range = range(x, 0, -1)
    else:
        test_range = range(x, image_width)
    
    for i in test_range:
        x1 = i
        try:
            y1 = int(x1 * perpendicular_slope + intercept)
        except ValueError:
            print(perpendicular_slope)
            raise

        # Distância euclidiana simples 
        distance = math.sqrt(((x - x1) ** 2) + ((y - y1) ** 2))
        current_distance = abs(desired_line_size - distance)

        # Cláusula de parada, já que a partir daqui, à medida que i cresce,
        # a tendência é que a distância apenas aumente...
        # Como não sabemos à priori quando isso deve ocorrer, setamos o range do
        # loop propositalmente alto.
        # Foi mais simples e "seguro" do que se basear na inclinação da reta
        # (que seria o ideal, para estimar melhor o ponto desejado)
        if current_distance > (desired_line_size * 1.10):
            break

        if closest_distance > current_distance:
            point = (x1, y1)
            closest_distance = current_distance

    return point


def lines_approximation(x1, y1, x2, y2, width):
    """
    Nos casos em que identificamos apenas a bottom_line,
    podemos supor duas linhas, já que nos frames anteriores,
    essas coordenadas existiam.
    """

    desired_size = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

    slope, intercept, _, _, _ = linregress([x1, x2], [y1, y2])
    
    if slope != 0:
        left_line = get_perpendicular_line(x1, y1, slope, width, desired_size)
        right_line = get_perpendicular_line(x2, y2, slope, width, desired_size)
    else:
        left_line = (x1, y1 - int(desired_size))
        right_line = (x2, y2 - int(desired_size))

    return left_line, right_line

'''
if __name__ == "__main__":
    width = 480
    height = 640
    img = np.zeros((width, height, 3), dtype=np.uint8)

    x1 = 100
    y1 = 80
    A = (x1, y1)

    x2 = 80
    y2 = 100
    B = (x2, y2)

    cv2.line(img, A, B, WHITE, 2)

    left_line, right_line = lines_approximation(x1, y1, x2, y2, width)

    cv2.line(img, (x1, y1), (left_line[0], left_line[1]), RED, 2)
    cv2.line(img, (x2, y2), (right_line[0], right_line[1]), RED, 2)

    cv2.imshow("Teste", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''