from datetime import datetime
import time

import cv2
import numpy as np

from .colors import BLUE, GOLD, RED
from .detector import hough_detection, match_line_with_shape, scale_detection
from .mask import Line, Shape, get_mask_coordinates
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
    coords_fallback = False
    TIME_COUNTER = 1
    LAST_DETECTION = 0
    MIN_LINE_LENGHT = 40
    DEVIATION_ALLOWED = 25

    start_time = datetime.now()

    # A cada frame processado
    while True:
        ret, frame = cap.read()

        # Giramos o frame, porque ele vem espelhado na captura
        frame = cv2.flip(frame, 1)
            
        hsv = frame_to_hsv(frame)
        edges = hsv_edges(hsv)

        TIME_COUNTER = TIME_COUNTER + 1

        # Na primeira iteração, esse valor é verdadeiro
        # Precisamos disso apenas porque somente dentro do laço
        # vamos ter acesso ao frame, para executar a get_mask_coordinates
        # Ou seja, SÓ RODA NA PRIMEIRA ITERAÇÃO!
        if set_mask_coords:
            mask_coords = get_mask_coordinates(frame)
            coords = get_mask_coordinates(frame)
            set_mask_coords = False

        lines = hough_detection(frame, edges, MIN_LINE_LENGHT)

        if not Shape.detected():
            for line in lines:
                # Na classe Shape é que vamos setar o que foi encontrado!
                # Como ela só tem atributos de classe, não estamos instanciando um objeto,
                # então não vai "perder" os valores entre um frame e outro                
                match_line_with_shape(frame, line, coords, Shape, DEVIATION_ALLOWED)
                if Shape.detected():
                    print('detectou direto')
                # cv2.line(frame, line.start, line.end, GREEN, 10)

        else:
            coords = [
                Line(Shape.left.start, Shape.left.end), 
                Line(Shape.right.start, Shape.right.end),
                Line(Shape.bottom.start, Shape.bottom.end),
            ]

            # https://stackoverflow.com/questions/2612802/how-to-clone-or-copy-a-list
            # Quando você atribui uma lista, Python só passa o ponteiro para ela
            # Para copiar para uma nova lista, tem que usar o .copy() sempre!
            aux = coords.copy()
            print(f"Entrou no deslocamento: {datetime.now().strftime('%H:%M:%S')}")
            print('Shape:\t', Shape.left, Shape.right, Shape.bottom)
            Shape.reset()

            PIXELS_VARIATION = 5

            for i in range(len(aux)):
                aux[i].x1 -= PIXELS_VARIATION
                aux[i].x2 -= PIXELS_VARIATION

            print('Aux:\t',aux[0], aux[1], aux[2])

            for line in lines:
                match_line_with_shape(frame, line, aux, Shape, DEVIATION_ALLOWED)

            print('Shape:\t', Shape.left, Shape.right, Shape.bottom)
            if Shape.detected():
                print('Deslocamento funcionou')
            # TODO: força bruta de Hough aqui
            # Manipular coords de acordo com a necessidade
            # Salvar o coords em uma auxiliar
            # 
            # aux = coords
            # 
            # 1. Deslocou pra esquerda:
            #   a. Setar o Shape.left, Shape.right e Shape.bottom pra None
            #   b. Diminuir o x1 e x2 de aux[0], aux[1] e aux[2] em N pixels (move todas pra esquerda)
            #   c. Chamar match_line_with_shape(frame, line, aux, Shape)
            # 
            # 2. Se Shape.detected(), sai do else, se não, aux = coords para resetar
            # 
            # 3. Deslocou pra direita
            #   a. Setar o Shape.left, Shape.right e Shape.bottom pra None
            #   b. Incrementar o x1 e x2 de aux[0], aux[1] e aux[2] em N pixels (move todas pra direita)
            #   c. Chamar match_line_with_shape(frame, line, aux, Shape)
            # 
            # 4. Se Shape.detected(), sai do else, se não, aux = coords para resetar
            # 
            # 5. Deslocou pra baixo
            #   a. Setar o Shape.left, Shape.right e Shape.bottom pra None
            #   b. Incrementar o y1 e y2 de aux[0], aux[1] e aux[2] em N pixels (move todas pra baixo)
            #   c. Chamar match_line_with_shape(frame, line, aux, Shape)
            #
            # 6. Se Shape.detected(), sai do else, se não, aux = coords para resetar
            #
            # 7. Deslocou pra cima
            #   a. Setar o Shape.left, Shape.right e Shape.bottom pra None
            #   b. Diminuir o y1 e y2 de aux[0], aux[1] e aux[2] em N pixels (move todas pra cima)
            #   c. Chamar match_line_with_shape(frame, line, aux, Shape)
            # 
            # 8. Se Shape.detected(), sai do else, se não, aux = coords para resetar
            # 
            # 9. Aumentou a escala
            #   a. Setar o Shape.left, Shape.right e Shape.bottom pra None
            #   b. Diminuir o y1 (cima) e incrementar o y2 (baixo) de aux[0] e aux[1] (left e right) em N pixels (aumenta o comprimento das linhas verticais)
            #       I. Isso é possível porque y1 sempre será o MENOR valor da coordenada em Y da linha, ou seja, o valor mais próximo da margem superior da imagem
            #       II. E, consequentemente, y2 sempre será o MAIOR valor da coordenada em Y da linha, ou seja, o valor mais próximo da margem inferior da imagem
            #       III. Ver o construtor da classe Line, em mask.py. Tem um if antes de setar os valores, pra sempre colocar em x1/y1 o menor valor e em x2/y2 o maior!
            #   c. Diminuir o x1 e x2 de aux[0] em N pixels (move a left pra esquerda)
            #   d. Incrementar o x1 e x2 de aux[1] em N pixels (move a right pra direita)
            #   e. Incrementar o x1 e x2 de aux[2] em N pixels (aumenta o comprimento da bottom)
            #   f. Incrementar o y1 e y2 de aux[2] em N pixels (move a bottom pra baixo)
            #   g. Chamar match_line_with_shape(frame, line, aux, Shape)
            #
            # 10. Se Shape.detected(), sai do else, se não, aux = coords para resetar
            #
            # 11. Diminuiu a escala
            #   a. Setar o Shape.left, Shape.right e Shape.bottom pra None
            #   b. Incrementar o y1 (cima) e diminuir o y2 (baixo) de aux[0] e aux[1] (left e right) em N pixels (diminui o comprimento das linhas verticais)
            #   c. Incrementar o x1 e x2 de aux[0] (left) em N pixels (move a left pra direita)
            #   d. Diminuir o x1 e x2 de aux[1] (right) em N pixels (move a right pra esquerda)
            #   e. Diminuir o x1 e x2 de aux[2] em N pixels (diminui o comprimento da bottom)
            #   f. Diminuir o y1 e y2 de aux[2] em N pixels (move a bottom pra cima)
            #   g. Chamar match_line_with_shape(frame, line, aux, Shape)
            #   
            # Um dos problemas: checar (com print) se aux está resetando para o valor original de coords
            # Em alguns casos, quando se faz a associação (aux = coords), pode ser que a variável receba um ponteiro... Python + C, não lembro os casos!
            # Como coords é associada a uma nova lista depois de chamar draw_tattoo() abaixo, dessa forma:
            #   coords = [Shape.left, Shape.right, Shape.bottom]
            # Isso não deveria acontecer. A forma como estou criando a lista é para que sempre seja uma nova área de memória sempre!
            # 
            # Diagonal??? Rotação???
            #   
            #   Diagonal poderia fazer passos similares, manipulando as coordenadas e usando match_line_with_shape
            #   
            #   Rotação precisa de outra ideia, e a match_line_with_shape tem aquelas verificações de line.vertical() e line.horizontal().
            #   Talvez role de usar algo parecido, mas aí usando esses métodos para excluir linhas verticais e horizontais do teste.
            #   Um outro problema com a rotação é como vamos desenhar a tatuagem.
            #   Usando o Geogebra porque fica melhor de visualizar, na imagem só vai ser "invertido" né, porque está de ponta cabeça.

            #   Por exemplo, uma linha esquerda rotacionada poderia ser: [(100, 100), (140, 20)]
            #   Ou seja, uma linha que tem 80 pixels de "comprimento" abs(y2 - y1), e está assim: \

            #   Digamos agora que a linha direita que combina poderia ser: [(160, 140), (200, 60)]
            #   Mesmo "comprimento", abs(y2 - y1) = 80 (na verdade não é, porque tem a inclinação)
            # 
            #   E a de baixo, que "fecha" com elas, seria: [(140, 20), (200, 60)]
            #   https://i.imgur.com/UyE5YMI.png
            #   
            #   COMO A GENTE VAI ITERAR NISSO??? Tem que iterar na diagonal!
            #
            #   Porque o ponto (0, 0) da tatuagem tem que ser escrito em (100, 100) -> de novo, usando as coordenadas do Geogebra pra facilitar
            #   Já o ponto (0, 1) tem que ser escrito em... (101, 101)??? depende da inclinação. Se ela estiver rotacionada 90 graus é fácil...
            #   Caso contrário... https://www.geeksforgeeks.org/program-find-line-passing-2-points/
            #   No exemplo ali do Geogebra, a solução seria f(x, y) = 40x - 60y, ou em equação de reta: y = (40x + 2000) / 60
            #   Quando passamos para essa fórmula os valores (100, 100) no Geogebra (começo do shape encontrado, parte esquerda superior), temos isso:
            #   https://i.imgur.com/b2yZgwW.png
            #
            #   Ou seja, essa linha laranja tem que ser seguida na primeira iteração. Como dá pra ver, Y e X não estão variando linearmente!
            #   Como a inclinação é positiva, sabemos que precisa ir diminuindo o Y (agora pensando nos termos da imagem mesmo) à medida que aumenta o X
            #   Porém, nesse caso, quando vamos um pixel pra direita:
            #   https://i.imgur.com/TrAOqLL.png
            #
            #   Tem que fazer arredondamentos. Porque quando X = 101, Y = 100.6 ali no Geogebra.
            #   Ou seja... vai ser MUITO mais difícil trabalhar com rotação!

        # Desenha o shape inicial na imagem capturada
        if Shape.detected():
            print('Desenhou shape direto')
            LAST_DETECTION += 1
            draw_tattoo(frame, Shape)
            coords = [
                Line(Shape.left.start, Shape.left.end),
                Line(Shape.right.start, Shape.right.end), 
                Line(Shape.bottom.start, Shape.bottom.end),
            ]
            # Shape.reset()
        elif LAST_DETECTION < 5 and coords[0] != mask_coords[0] and coords[1] != mask_coords[1] and coords[2] != mask_coords[2]:
            LAST_DETECTION += 1
            Shape.set_coords(coords)
            draw_tattoo(frame, Shape)
            Shape.reset()
        elif TIME_COUNTER % 50 == 0:
            print(f"RESETOU TIMER: {datetime.now().strftime('%H:%M:%S')}")
            LAST_DETECTION = 0
            Shape.reset()
            coords = mask_coords
            draw_lines(frame, mask_coords, BLUE) 
        else:
            LAST_DETECTION = 0
            coords = mask_coords
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
