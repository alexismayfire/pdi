import cv2
import numpy as np
import time


SQUARE = "SQUARE"
X = "X"


cap = cv2.VideoCapture(0)

# Confere se a webcam pode ser acessada
if not cap.isOpened():
    raise IOError("Cannot open webcam")


class Mask:
    def __init__(self, shape, frame, lines):
        self.shape = shape
        self.frame = frame
        self.lines = lines
        height, width, _ = frame.shape
        x = width // 3  # 33% da largura
        self.x = int(x + (x // 5))
        self.y = height // 3  # 33% da altura
        self.height = height
        self.width = width

    def _square_shape(self):
        # Linha da esquerda
        cv2.line(
            self.frame, (self.x, self.y), (self.x, self.height - self.y), (255, 0, 0), 3
        )
        # Linha da direita
        cv2.line(
            self.frame,
            (self.width - self.x, self.y),
            (self.width - self.x, self.height - self.y),
            (255, 0, 0),
            3,
        )
        # Linha inferior
        cv2.line(
            self.frame,
            (self.x + 10, self.height - self.y),
            (self.width - self.x - 10, self.height - self.y),
            (255, 0, 0),
            3,
        )

    def _detect_square(self, deviation_allowed=5):
        # TODO: por enquanto só detecta linhas horizontais e verticais, cortando
        # as diagonais. Linhas inclinadas são aceitas desde que a inclinação
        # esteja dentro do parâmetro de desvio

        if self.lines is not None:
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                if deviation_allowed > abs(x1 - x2) or deviation_allowed > abs(y1 - y2):
                    # Aqui ainda seria necessário um teste, talvez armazenar em um auxiliar
                    # as linhas de interesse.
                    #
                    # De novo, caso seja fixado um 'crosshair' na tela com o símbolo onde
                    # a tatuagem será aplicada, vamos saber à priori qual deve ser a distância
                    # entre as duas linhas verticais, e uma ideia de y onde a linha horizontal
                    # deve estar (bem próxima ao maior valor de x1 ou x2 de qualquer
                    # uma das linhas verticais)
                    cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    def _x_shape(self):
        # Linha da esquerda
        cv2.line(
            self.frame,
            (self.x, self.y),
            (self.width - self.x, self.height - self.y),
            (255, 0, 0),
            3,
        )
        # Linha da direita
        cv2.line(
            self.frame,
            (self.width - self.x, self.y),
            (self.x, self.height - self.y),
            (255, 0, 0),
            3,
        )

    def _detect_x(self):
        pass

    def detect(self):
        if self.shape == SQUARE:
            self._square_shape()
            self._detect_square()
        else:
            self._x_shape()
            self._detect_x()

        return self.frame


def detect_x_shape(frame, line):
    x1, y1, x2, y2 = line[0]


def hough(frame, edges, format=SQUARE):
    # Aqui tentando identificar aquele mesmo símbolo do InkHunter: |_|
    # Thresh acima de 100 ja dificulta pra encontrar
    thresh = 80
    # Esse valor aparentemente é fixo no InkHunter, se você não encaixar
    # certinho na sobreposição do símbolo não acha.
    # Então precisa definir um valor bacana aqui pra ajudar no Hough!
    min_line_length = 10
    # Um line gap maior exibe melhor somente as linhas que interessam.
    # Se diminuir e tiver uma "falha" no Canny (ou qualquer outro algoritmo
    # usado pra detectar as bordas), ou seja, descontinuidade de borda,
    # vai prejudicar o reconhecimento.
    max_line_gap = 5
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, thresh, None, min_line_length, max_line_gap
    )

    return lines


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

    edges = cv2.Canny(gray, 120, 200)

    lines = hough(frame, edges)
    mask = Mask(SQUARE, frame, lines)
    frame = mask.detect()
    """
    height, width, channels = frame.shape
    x = width // 3  # 33% da largura
    y = height // 3  # 33% da altura
    y1 = (height // 2) + y
    y2 = (height // 2) - y

    x = int(x + (x // 5))

    # Linha da esquerda
    cv2.line(frame, (x, y), (x, height - y), (255, 0, 0), 3)
    # Linha da direita
    cv2.line(frame, (width - x, y), (width - x, height - y), (255, 0, 0), 3)
    # Linha inferior
    cv2.line(frame, (x + 10, height - y), (width - x - 10, height - y), (255, 0, 0), 3)

    # Linha da esquerda
    cv2.line(frame, (x, y), (width - x, height - y), (255, 0, 0), 3)
    # Linha da direita
    cv2.line(frame, (width - x, y), (x, height - y), (255, 0, 0), 3)
    """
    height, width, channels = frame.shape
    side = np.zeros((height, width * 2, channels), dtype=np.uint8)
    side[:, :width, :] = frame[:, :, :]
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
