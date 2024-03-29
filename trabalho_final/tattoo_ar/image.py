import cv2
import numpy as np

from .colors import GOLD, BLUE, RED
from .mask import Line, Shape
from .line import get_perpendicular_line, lines_approximation


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
    width, height, _ = hsv.shape
    binarized = np.where(hsv>=np.array([10,10,0]),255,0)
    return binarized[:,:,0].astype(np.uint8)
    # return cv2.Canny(hsv, threshold_1, threshold_2)


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
    frame_y_range = None
    frame_x_range = None
    size = None

    # Se todas as linhas do shape foram encontradas
    if type(shape) == Shape and shape.left and shape.right and shape.bottom:
        # lines = [shape.left, shape.right, shape.bottom]
        # draw_lines(frame, lines, GOLD)

        frame_y_range = range(Shape.y_start(), Shape.y_end())
        frame_x_range = range(Shape.left.x, Shape.right.x)
        size = shape.size()

    elif type(shape) == list:
        y_start = (shape[0].y1 + shape[1].y1) // 2
        vertical_y = (shape[0].y2 + shape[1].y2) // 2
        if vertical_y > shape[2].y:
            y_end = vertical_y
        else:
            y_end = shape[2].y
            
        frame_y_range = range(y_start, y_end)
        frame_x_range = range(shape[0].x, shape[1].x)

        width = abs(shape[1].x - shape[0].x)

        if shape[1].y1 > shape[0].y1:
            height = abs(shape[1].y1 - shape[2].y2)
        else:
            height = abs(shape[0].y1 - shape[2].y2)
        
        size = (width, height)

    if frame_y_range and frame_x_range:
        tattoo = tattoo_image(size)
        tattoo_width, tattoo_height = tattoo.shape
        tattoo_y_range = range(tattoo_width)
        tattoo_x_range = range(tattoo_height)

        for y, y2 in zip(frame_y_range, tattoo_y_range):
            for x, x2 in zip(frame_x_range, tattoo_x_range):
                if tattoo[y2, x2] == 0:
                    frame[y, x] = np.array([
                        tattoo[y2, x2], tattoo[y2, x2], tattoo[y2, x2]
                    ])

def draw_tattoo_somehow(frame, all_lines):

    # Left , Rigth, Bottom
    # print('All Lines:', all_lines[0], '|',all_lines[1], '|',all_lines[2])

    aux = all_lines.copy()

    # Tamanho da linha Left
    if aux[0] is not None:
        left_line_size = int(np.linalg.norm(np.array((Shape.left.x1, Shape.left.y1)) - np.array((Shape.left.x2, Shape.left.y2)))) 
    
    # Tamanho da linha Right
    if aux[1] is not None:
        right_line_size = int(np.linalg.norm(np.array((Shape.right.x1, Shape.right.y1)) - np.array((Shape.right.x2, Shape.right.y2)))) 

    # Tamanho da linha Bottom
    if aux[2] is not None:
        bottom_line_size = int(np.linalg.norm(np.array((Shape.bottom.x1, Shape.bottom.y1)) - np.array((Shape.bottom.x2, Shape.bottom.y2)))) 
    
    # Só a Bottom
    if aux[0] is None and aux[1] is None and aux[2] is not None:
        print('Só bottom')          
        #aux[0] = Line((aux[2].x1, aux[2].y1), (aux[2].x1, aux[2].y1 - bottom_line_size))
        #aux[1] = Line((aux[2].x2, aux[2].y2), (aux[2].x2, aux[2].y2 - bottom_line_size))
        if aux[2].x1 == aux[2].x2:
            aux[2].x1 + 3
        if aux[2].y1 == aux[2].y2:
            aux[2].y1 + 3
        cv2.line(frame, aux[2].start, aux[2].end, (255,200,50), 15)
        
        #width = 480
        #height = 640
        #img = np.zeros((width, height, 3), dtype=np.uint8)

        #x1 = 100
        #y1 = 80
        x1 = aux[2].x1
        y1 = aux[2].y1
        A = (x1, y1)

        #x2 = 80
        #y2 = 100
        x2 = aux[2].x2
        y2 = aux[2].y2
        B = (x2, y2)

        # Desenhando
        cv2.line(frame, A, B, (45,45,45), 2)

        left_line, right_line = lines_approximation(x1, y1, x2, y2, 480)
 
        # Linhas Inferidas
        # new_line_left = Line((x1, y1), (left_line[0], left_line[1]))
        # print(new_line_left)

        cv2.line(frame, (x1, y1), (left_line[0], left_line[1]), RED, 2)
        cv2.line(frame, (x2, y2), (right_line[0], right_line[1]), RED, 2)

        aux[0] = Line((x1, y1), (left_line[0], left_line[1]))
        aux[1] = Line((x2, y2), (right_line[0], right_line[1]))
        cv2.line(frame, aux[0].start, aux[0].end, (50, 255, 0), 2)
        cv2.line(frame, aux[1].start, aux[1].end, (50, 255, 0), 2)

        # Linhas Inferidas
        #cv2.line(frame, aux[0].start, aux[0].end, (255,200,50), 15)
        #cv2.line(frame, aux[1].start, aux[1].end, (255,200,50), 15)

    # Só a Right
    if aux[0] is None and aux[1] is not None and aux[2] is None:
        print('Só Right') 
        aux[2] = Line((aux[1].x2 - right_line_size, aux[1].y2), (aux[1].x2, aux[1].y2))
        aux[0] = Line((aux[2].x1, aux[2].y1), (aux[2].x1, aux[2].y1 - right_line_size))

        # Linhas Inferidas
        # cv2.line(frame, aux[2].start, aux[2].end, (255,200,50), 15)
        # cv2.line(frame, aux[0].start, aux[0].end, (255,200,50), 15)

    # Só a Left
    if aux[0] is not None and aux[1] is None and aux[2] is None: 
        print('Só Left') 
        aux[2] = Line((aux[0].x2, aux[0].y2), (aux[0].x2 + left_line_size, aux[0].y2))
        aux[1] = Line((aux[2].x2, aux[2].y2), (aux[2].x2, aux[2].y2 - left_line_size))

        # Linhas Inferidas
        # cv2.line(frame, aux[2].start, aux[2].end, (255,200,50), 15)
        # cv2.line(frame, aux[1].start, aux[1].end, (255,200,50), 15)

    # Bottom e Left
    if aux[0] is not None and aux[1] is None and aux[2] is not None:  
        aux[1] = Line((aux[2].x2, aux[2].y2), (aux[2].x2, aux[2].y2 - left_line_size))

    # Bottom e Right
    if aux[0] is None and aux[1] is not None and aux[2] is not None:  
        aux[0] = Line((aux[2].x1, aux[2].y1), (aux[2].x1, aux[2].y1 - right_line_size))

    # Left e Right
    if aux[0] is not None and aux[1] is not None and aux[2] is None:  
        aux[2] = Line((aux[0].x2, aux[0].y2), (aux[1].x2, aux[1].y2))

    # print('Bottom', aux[2], 'Nova Esquerda:', aux[0], 'Nova Direita:', aux[1])
    Shape.left = aux[0]
    Shape.right = aux[1]
    Shape.bottom = aux[2]

    '''    
    # Ponto Médio
    Line.x = (self.x1 + self.x2) // 2
    Line.y = (self.y1 + self.y2) // 2

    # Inclinação
    Line.x_deviation = abs(self.x1 - self.x2)
    Line.y_deviation = abs(self.y1 - self.y2)

    # aux[2].horizontal(max_deviation=15) -> retorna True se for linha horizontal
    # aux[2].vertical(max_deviation=15) -> retorna True se for vertical

    '''   

