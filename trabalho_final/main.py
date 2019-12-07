import cv2
import numpy as np
import time


SQUARE = "SQUARE"
X = "X"
LEFT_LINE = None
RIGHT_LINE = None
BOTTOM_LINE = None
TIME_COUNTER = 1
LEFT_MASK_LINE = [(255,160),(255,320)]
RIGHT_MASK_LINE = [(385,160),(385,320)]
BOTTOM_MASK_LINE = [(265,320),(375,320)]


class Mask:
    def __init__(self, shape, frame, lines, left_line, right_line, bottom_line):
        self.shape = shape
        self.frame = frame
        self.lines = lines
        height, width, _ = frame.shape  # tamanho da tela
        x = width // 3  # 33% da largura
        self.x = int(x + (x // 5))
        self.y = height // 3  # 33% da altura
        self.height = height  # 480
        self.width = width  # 640
        self.left_line = left_line
        self.right_line = right_line
        self.bottom_line = bottom_line

        # Posição da imagem
        self.s_img = cv2.imread("tattoo1.png")
        self.x_offset = 280
        self.y_offset = 240 - self.s_img.shape[1]
        self.y1, self.y2 = self.y_offset, self.y_offset + self.s_img.shape[0]
        self.x1, self.x2 = self.x_offset, self.x_offset + self.s_img.shape[1]
        self.alpha_s = self.s_img[:, :, 0] / 255
   
    def _square_shape(self):  # desenha a forma de 3 retas no meio da tela
        color = (255, 0, 0)
        left_start, left_end = (255, 160), (255, 320)
        right_start, right_end = (385, 160), (385, 320)
        bottom_start, bottom_end = (265, 320), (375, 320)

        cv2.line(self.frame, left_start, left_end, color, 3)
        cv2.line(self.frame, right_start, right_end, color, 3)
        cv2.line(self.frame, bottom_start, bottom_end, color, 3)

    def _detect_square(self, deviation_allowed=15):
        # TODO: por enquanto só detecta linhas horizontais e verticais, cortando
        # as diagonais. Linhas inclinadas são aceitas desde que a inclinação
        # esteja dentro do parâmetro de desvio deviation_allowed
        
        # Analisa as linhas
        if self.lines is not None:
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 3) Imprime todas as linhas

                # Linhas verticais , com tolerância em deviation
                if deviation_allowed > abs(x1 - x2) and deviation_allowed < abs(
                    y1 - y2
                ):
                    # De novo, caso seja fixado um 'crosshair' na tela com o símbolo onde
                    # a tatuagem será aplicada, vamos saber à priori qual deve ser a distância
                    # entre as duas linhas verticais, e uma ideia de y onde a linha horizontal
                    # deve estar (bem próxima ao maior valor de x1 ou x2 de qualquer
                    # uma das linhas verticais)

                    # LINHA DA ESQUERDA
                    left_x = 255
                    vertical_lower_y = 160
                    vertical_upper_y = 320
                    l_lower_x = left_x - deviation_allowed
                    l_upper_x = left_x + deviation_allowed

                    right_x = 385
                    r_lower_x = right_x - deviation_allowed
                    r_upper_x = right_x + deviation_allowed

                    if (l_lower_x < x1 < l_upper_x) or (l_lower_x < x2 < l_upper_x):
                        # (self.x, self.y), (self.x, self.height - self.y)

                        dif_x1 = abs(x1 - left_x)
                        dif_x2 = abs(x2 - left_x)

                        # Pega a menor diferença, caso a linha esteja ligeiramente inclinada
                        if dif_x1 > dif_x2:
                            dif_x = dif_x2
                        else:
                            dif_x = dif_x1

                        # Identifica se está próximo da linha do centro da esquerda?
                        if y1 > y2:
                            dif_y = abs(y2 - vertical_lower_y) + abs(y1 - vertical_upper_y)
                        else:
                            dif_y = abs(y1 - vertical_lower_y) + abs(y2 - vertical_upper_y)

                        # O quanto pode estar distante da linha de marcação na tela (Tolerância)
                        tolerance_y = 25
                        tolerance_x = 25

                        if (
                            tolerance_y > dif_y
                            and tolerance_x > dif_x
                            and self.left_line is None
                        ):
                            self.left_line = [(x1, y1), (x2, y2)]
                        else:
                            cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    elif (r_lower_x < x1 < r_upper_x) or (r_lower_x < x2 < r_upper_x):
                        # LINHA DA DIREITA
                        # (RIGHT_MASK_LINE[0][0], self.y), (RIGHT_MASK_LINE[0][0], self.height - self.y)
                        dif_x1 = abs(x1 - right_x)
                        dif_x2 = abs(x2 - right_x)

                        # Pega a menor diferença, caso a linha esteja ligeiramente inclinada
                        if dif_x1 > dif_x2:
                            dif_x = dif_x2
                        else:
                            dif_x = dif_x1

                        # Identifica se está próximo da linha do centro da direita
                        if y1 > y2:
                            dif_y = abs(y2 - vertical_lower_y) + abs(y1 - vertical_upper_y)
                        else:
                            dif_y = abs(y1 - vertical_lower_y) + abs(y2 - vertical_upper_y)

                        # O quanto pode estar distante da linha de marcação na tela (Tolerância)
                        tolerance_y = 25
                        tolerance_x = 25

                        if (
                            tolerance_y > dif_y
                            and tolerance_x > dif_x
                            and self.right_line is None
                        ):
                            self.right_line = [(x1, y1), (x2, y2)]

                        else:
                            cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # Linha horizontal
                elif deviation_allowed > abs(y1 - y2) and deviation_allowed < abs(
                    x1 - x2
                ):
                    target_y = 320 
                    horizontal_lower_x = 265
                    horizontal_upper_x = 375
                
                    target_lower_y = target_y - deviation_allowed
                    target_upper_y = target_y + deviation_allowed

                    if (target_lower_y < y1 < target_upper_y) or (target_lower_y < y2 < target_upper_y):
                        dif_y1 = abs(y1 - target_y)
                        dif_y2 = abs(y2 - target_y)

                        # Pega a menor diferença, caso a linha esteja ligeiramente inclinada
                        if dif_y1 > dif_y2:
                            dif_y = dif_y2
                        else:
                            dif_y = dif_y1

                        # Identifica se está próximo da linha do centro da esquerda?
                        if x1 > x2:
                            dif_x = abs(x2 - horizontal_lower_x) + abs(x1 - horizontal_upper_x)
                        else:
                            dif_x = abs(x1 - horizontal_lower_x) + abs(x2 - horizontal_upper_x)

                        # O quanto pode estar distante da linha de marcação na tela (Tolerância)
                        tolerance_y = 25
                        tolerance_x = 25

                        if (
                            tolerance_y > dif_y
                            and tolerance_x > dif_x
                            and self.bottom_line is None
                        ):
                            self.bottom_line = [(x1, y1), (x2, y2)]
                        else:
                            cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                

    def _draw_tattoo(self):
        if self.left_line and self.right_line and self.bottom_line:
            # Left Line Params
            (x1, y1), (x2, y2) = self.left_line
            cv2.line(self.frame, (x1, y1), (x2, y2), (0, 250, 160), 3)  # Yellow | Linha Amarela

            # Right Line Params
            (x1, y1), (x2, y2) = self.right_line
            cv2.line(self.frame, (x1, y1), (x2, y2), (0, 250, 160), 3)  # Yellow | Linha Amarela

            # Bottom Line Params
            (x1, y1), (x2, y2) = self.bottom_line
            cv2.line(self.frame, (x1, y1), (x2, y2), (0, 250, 160), 3)  # Yellow | Linha Amarela

            (x1, y1), (x2, y2) = self.left_line
            x_left = (x1 + x2) // 2 # Valor médio do X da linha da esquerda
            y_left = (y1 + y2) // 2 # Valor médio do Y da linha da esquerda

            (x1, y1), (x2, y2) = self.right_line
            x_right = (x1 + x2) // 2 # Valor médio do X da linha da direita
            y_right = (y1 + y2) // 2 # Valor médio do Y da linha da direita
        
            (x1, y1), (x2, y2) = self.bottom_line

            y_lower = y_right if y_right > y_left else y_left
            y_upper = (y1 + y2) // 2 # Valor médio do Y na linha de baixo

            y1_aux, y2_aux = (y_lower, y_upper)
            x1_aux, x2_aux = (x_left, x_right)

            # for c in range(0, 2):
            # self.frame[y1_aux:y2_aux, x1_aux:x2_aux] = (self.alpha_s * self.s_img[:, :] + self.alpha_l * self.frame[y1_aux:y2_aux, x1_aux:x2_aux])
            for y, y2 in zip(range(y1_aux, y2_aux), range(0, self.alpha_s.shape[0])):
                for x, x2 in zip(range(x1_aux, x2_aux), range(0, self.alpha_s.shape[1])):
                    if self.alpha_s[y2, x2] == 0:
                        self.frame[y, x, 0] = self.alpha_s[y2, x2]
                        self.frame[y, x, 1] = self.alpha_s[y2, x2]
                        self.frame[y, x, 2] = self.alpha_s[y2, x2]

    def detect(self):
        if self.shape == SQUARE:
            self._square_shape()
            self._detect_square()
            self._draw_tattoo()
        else:
            self._x_shape()
            self._detect_x()

        return (self.frame, self.left_line, self.right_line, self.bottom_line)


def hough(frame, edges, format=SQUARE):
    # Aqui tentando identificar aquele mesmo símbolo do InkHunter: |_|
    # Thresh acima de 100 ja dificulta pra encontrar
    thresh = 40
    # Esse valor aparentemente é fixo no InkHunter, se você não encaixar
    # certinho na sobreposição do símbolo não acha.
    # Então precisa definir um valor bacana aqui pra ajudar no Hough!
    min_line_length = 80
    # Um line gap maior exibe melhor somente as linhas que interessam.
    # Se diminuir e tiver uma "falha" no Canny (ou qualquer outro algoritmo
    # usado pra detectar as bordas), ou seja, descontinuidade de borda,
    # vai prejudicar o reconhecimento.
    max_line_gap = 10
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, thresh, None, min_line_length, max_line_gap
    )

    return lines


def hsv_blue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    background = np.zeros(frame.shape, dtype=np.uint8)
    sensitivity = 20
    lower_range = np.array(
        [110 - sensitivity, 50, 50]
    )  # Range de cores analisadas (limite inferior)
    upper_range = np.array(
        [130 + sensitivity, 255, 255]
    )  # Range de cores analisadas (limite inferior)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask_3d = np.zeros(frame.shape, dtype=np.uint8)
    for ch in range(2):
        mask_3d[:, :, ch] = mask[:, :]
    pre_edges = np.where(mask_3d == 0, background, frame)

    return pre_edges


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # Confere se a webcam pode ser acessada
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # A cada frame
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, 1)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # sensitividade = 20
        # lower = np.array([120 - sensitividade, 60, 30])
        # upper = np.array([120 + sensitividade, 255, 255])
        # mask = cv2.inRange(hsv, lower, upper)

        # gray = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        # gray[:, :] = np.where(mask == 0, 0, 255)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pre_edges = hsv_blue(frame)

        edges = cv2.Canny(pre_edges, 80, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        lines = hough(frame, edges)
        mask = Mask(SQUARE, frame, lines, LEFT_LINE, RIGHT_LINE, BOTTOM_LINE)

        TIME_COUNTER = TIME_COUNTER + 1
     
         #   print(LEFT_LINE, BOTTOM_LINE)
        '''
        if LEFT_LINE is None:
            print(TIME_COUNTER)
        else:
            print(TIME_COUNTER, LEFT_LINE)
        '''

        frame, left_line, right_line, bottom_line = mask.detect()
        if LEFT_LINE is None:
            LEFT_LINE = left_line
        if RIGHT_LINE is None:
            RIGHT_LINE = right_line
        if BOTTOM_LINE is None:
            BOTTOM_LINE = bottom_line

         # A cada 10 frames zera as linhas e recomeça a analisar
        if TIME_COUNTER % 20 == 0:
            LEFT_LINE = None
            RIGHT_LINE = None
            BOTTOM_LINE = None

        print('---------------')
        print(f'LEFT: {left_line}')
        print(f'RIGHT: {right_line}')
        print(f'BOTTOM:{bottom_line}')
        print('---------------')

        height, width, channels = frame.shape
        side = np.zeros((height, width * 2, channels), dtype=np.uint8)
        side[:, :width, :] = frame[:, :, :]
        side[:, width:, :] = pre_edges[:, :, :]
        # side[:, width:, 0] = edges[:, :]
        # side[:, width:, 1] = edges[:, :]
        # side[:, width:, 2] = edges[:, :]
        cv2.imshow("Input", side)

        c = cv2.waitKey(1)
        # Valor ASCII do Esc
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
