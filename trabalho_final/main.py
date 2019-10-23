import cv2
import numpy as np


cap = cv2.VideoCapture(0)

# Confere se a webcam pode ser acessada
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 75)

    # Aqui tentando identificar aquele mesmo símbolo do InkHunter: |_|
    # Thresh acima de 100 ja dificulta pra encontrar
    thresh = 60
    # Esse valor aparentemente é fixo no InkHunter, se você não encaixar
    # certinho na sobreposição do símbolo não acha.
    # Então precisa definir um valor bacana aqui pra ajudar no Hough!
    min_line_length = 50
    # Um line gap maior exibe melhor somente as linhas que interessam.
    # Se diminuir e tiver uma "falha" no Canny (ou qualquer outro algoritmo
    # usado pra detectar as bordas), ou seja, descontinuidade de borda,
    # vai prejudicar o reconhecimento.
    max_line_gap = 30
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, thresh, None, min_line_length, max_line_gap
    )
    if lines is not None:
        # "lines" é uma lista com todas as linhas identificadas.
        # Precisa do teste acima pra não dar erro na hora de acessar o vetor
        # Pode ser que em um frame não detectou nada através dos parâmetros acima!
        for line in lines:
            # "line" é um array no formato do numpy, ou seja, [[x1, y1, x2, y2]]
            # Por isso precisa acessar o índice 0
            x1, y1, x2, y2 = line[0]
            # Aqui é para excluir linhas diagonais.
            # x1 e y1 são a coluna e linha inicias da linha
            # x2 e y2 são a coluna e linha finais
            # Quando a diferença absoluta de x1 e x2 é pequena, é uma linha horizontal
            # Quando a diferença absoluta de y1 e y2 é pequena, é uma linha vertical
            desvio_permitido = 3
            if desvio_permitido > abs(x1 - x2) or desvio_permitido > abs(y1 - y2):
                # Aqui ainda seria necessário um teste, talvez armazenar em um auxiliar
                # as linhas de interesse.
                #
                # De novo, caso seja fixado um 'crosshair' na tela com o símbolo onde
                # a tatuagem será aplicada, vamos saber à priori qual deve ser a distância
                # entre as duas linhas verticais, e uma ideia de y onde a linha horizontal
                # deve estar (bem próxima ao maior valor de x1 ou x2 de qualquer
                # uma das linhas verticais)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow("Input", frame)

    c = cv2.waitKey(1)
    # Valor ASCII do Esc
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
