from typing import List

import cv2
import numpy as np

from .colors import BLUE, GOLD, RED
from .image import draw_line
from .mask import Line


def hough_detection(frame, edges, min_line_length):
    # Aqui tentando identificar aquele mesmo símbolo do InkHunter: |_|
    # Thresh acima de 100 ja dificulta pra encontrar
    thresh = 40

    # Esse valor aparentemente é fixo no InkHunter, se você não encaixar
    # certinho na sobreposição do símbolo não acha.
    # Então precisa definir um valor bacana aqui pra ajudar no Hough!
    #min_line_length = 80

    # Um line gap maior exibe melhor somente as linhas que interessam.
    # Se diminuir e tiver uma "falha" no Canny (ou qualquer outro algoritmo
    # usado pra detectar as bordas), ou seja, descontinuidade de borda,
    # vai prejudicar o reconhecimento.
    max_line_gap = 10
    hough_lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, thresh, None, min_line_length, max_line_gap
    )

    lines = []
    if hough_lines is not None:
        for hough_line in hough_lines:
            x1, y1, x2, y2 = hough_line[0]
            lines.append(Line((x1, y1), (x2, y2)))

    return lines


def match_line_with_shape(frame, line: Line, target_coords: List[Line], matched_coords, max_deviation):
    left_line, right_line, bottom_line = target_coords

    # target_coords são as coordenadas que vão ser usadas para comparação
    if line.vertical(max_deviation):
        for target_line, orientation in zip([left_line, right_line], ['left', 'right']):
            lower_x = target_line.x1 - max_deviation
            upper_x = target_line.x2 + max_deviation

            lower_y = target_line.y1 - max_deviation
            upper_y = target_line.y2 + max_deviation

            if (lower_x < line.x1 < upper_x) or (lower_x < line.x2 < upper_x):
                dif_x1 = abs(line.x1 - target_line.x)
                dif_x2 = abs(line.x2 - target_line.x)

                if dif_x1 > dif_x2:
                    dif_x = dif_x2
                else:
                    dif_x = dif_x1

                if line.y1 > line.y2:
                    dif_y = abs(line.y2 - lower_y) + abs(line.y1 - upper_y)
                else:
                    dif_y = abs(line.y1 - lower_y) + abs(line.y2 - upper_y)

                tolerance_y = 25
                tolerance_x = 25
            
                # getattr vai buscar uma propriedade da classe Shape,
                # de acordo com o valor de orientation, com default de None
                shape_line = getattr(matched_coords, orientation, None)
                if (
                    tolerance_y > dif_y 
                    and tolerance_x > dif_x
                    and shape_line is None
                ):
                    # setattr vai setar a propriedade da classe Shape,
                    # de acordo com orientation (pode ser 'left' ou 'right'),
                    # depende de em qual iteração está
                    # Primeiro vai tentar ver se é a linha da esquerda,
                    # Se não der resultado, vai tentar ver se não é a da direita
                    setattr(matched_coords, orientation, line)
                elif shape_line is None:
                    #Desenha todas as linhas verticais
                    #draw_line(frame, line, RED)
                    pass

    elif line.horizontal(max_deviation):
        lower_x = bottom_line.x1
        upper_x = bottom_line.x2
        
        lower_y = bottom_line.y1 - max_deviation
        upper_y = bottom_line.y2 + max_deviation

        if (lower_y < line.y1 < upper_y) or (lower_y < line.y2 < upper_y):
            dif_y1 = abs(line.y1 - bottom_line.y)
            dif_y2 = abs(line.y2 - bottom_line.y)

            if dif_y1 > dif_y2:
                dif_y = dif_y2
            else:
                dif_y = dif_y1

            if line.x1 > line.x2:
                dif_x = abs(line.x2 - lower_x) + abs(line.x1 - upper_x)
            else:
                dif_x = abs(line.x1 - lower_x) + abs(line.x2 - upper_x)

            # Na linha horizontal, tolerance_y é quanto a linha encontrada pode estar
            # desalinhada, na horizontal, com o target desejado
            tolerance_y = 25
            # Na linha horizontal, tolerance_x pode representar o quanto aceitamos
            # que a linha não se enquadre na vertical
            # Ou seja, pode ser tanto uma linha menor que o target,
            # ex.: 10px a menos na esquerda e 12px a menos na direita, entraria!
            # ex2.: 10px a mais na esquerda e 12px a menos na direita, entraria!
            tolerance_x = 25
          
            # getattr vai buscar a propriedade 'bottom' da classe Shape
            # Mantive só para padronizar
            shape_line = getattr(matched_coords, 'bottom', None)
            if (
                tolerance_y > dif_y 
                and tolerance_x > dif_x
                and shape_line is None
            ):
                # setattr vai setar a propriedade da classe Shape,
                # de acordo com orientation
                setattr(matched_coords, 'bottom', line)
            elif shape_line is None:
                # Desenha todas as linhas horizontais 
                #draw_line(frame, line, RED)
                pass

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))