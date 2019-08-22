from datetime import datetime
import time

import cv2
import numpy as np

# Aqui usa as mesmas constantes que o código em C
INPUT_IMAGE = "arroz.bmp"
NEGATIVO = 0
THRESHOLD = 0.4
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 1

VERMELHO = np.array([0, 0, 1])


class Coordenada:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Retangulo:
    def __init__(self, cima, baixo, esquerda, direita):
        self.c = cima
        self.b = baixo
        self.e = esquerda
        self.d = direita


class Componente:
    def __init__(self, retangulo):
        self.retangulo = retangulo
        self.label = np.float(0)
        self.n_pixels = 0


def salvar_imagem(nome, img):
    # Antes de salvar, como abrimos em valores normalizados e float (valores entre [0, 1])
    # agora precisa multiplicar por 255, seguindo as orientações do professor

    # Syntactic sugar, como img.shape é um vetor com 3 posições, podemos acessar assim.
    # O equivalente seria:

    # altura = img.shape[0]
    # largura = img.shape[1]
    # canais = img.shape[2]
    altura, largura, canais = img.shape

    for y in range(0, altura):
        for x in range(0, largura):
            for canal in range(0, canais):
                img[y][x][canal] = img[y][x][canal] * 255

    cv2.imwrite(nome, img)


def inverte(img):
    # Explicação desse syntactic sugar na função salvar_imagem() acima!
    #
    # Mas aqui, eu defino "_" como boa prática, porque não precisamos desse valor.
    # Porém, precisa "desempacotar" do vetor img.shape! Se fizer apenas assim:
    #
    # altura, largura = img.shape
    #
    # Vai dar uma exceção:
    # ValueError: too many values to unpack (expected 2)
    # Porque definimos duas variáveis, mas img.shape é um vetor com 3 posições
    # e todas precisam ser retornadas
    altura, largura, _ = img.shape

    # Criamos uma nova matriz preenchida com zeros e do mesmo tamanho que a imagem
    img_invertida = np.zeros(img.shape)

    for y in range(0, altura):
        for x in range(0, largura):
            # Imagens abertas no OpenCV tem os canais invertidos
            # Ao invés de RGB, retorna BGR!
            # Outro syntactic sugar, como sabemos que "img[y][x]" é um vetor com 3 posições
            # (os valores dos 3 canais), podemos definir 3 variáveis, cada uma recebendo
            # o valor na ordem do vetor.
            # É equivalente a:
            # blue = img[y][x][0]
            # green = img[y][x][1]
            # red = img[y][x][2]
            b, g, r = img[y][x]

            # Isso aqui funciona corretamente apenas se a imagem for aberta normalizada!
            # Seguindo o código em C do professor, basta diminuir 1 dos valores dos canais
            img_invertida[y][x][0] = np.float(1) - b
            img_invertida[y][x][1] = np.float(1) - g
            img_invertida[y][x][2] = np.float(1) - r

    return img_invertida


def desenha_linha(p1, p2, img, cor):
    # Ver a explicação disso na função inverte() acima!
    altura, largura, _ = img.shape

    if p1.x == p2.x:
        # Linha vertical
        inicio = max(0, min(p1.y, p2.y))
        fim = min(altura - 1, max(p1.y, p2.y))

        for y in range(inicio, fim):
            img[y][p1.x] = cor

    elif p1.y == p2.y:
        # Linha horizontal
        inicio = max(0, min(p1.x, p2.x))
        fim = min(largura - 1, max(p1.x, p2.x))

        for x in range(inicio, fim):
            img[p1.y][x] = cor

    else:
        pass
        # raise NotImplementedError(
        #     "TODO: desenhaLinha: implementar para linhas inclinadas!"
        # )

    return img


def desenha_retangulo(r, img, cor=VERMELHO):
    # Ver a explicação disso na função inverte() acima!
    altura, largura, _ = img.shape

    # Essa função não foi comentada pelo professor, mas tem a seguinte lógica:

    # Esquerda.
    if r.e >= 0 and r.e < altura:
        img = desenha_linha(Coordenada(r.e, r.c), Coordenada(r.e, r.b), img, cor)

    # Direita.
    if r.d >= 0 and r.d < altura:
        img = desenha_linha(Coordenada(r.d, r.c), Coordenada(r.e, r.b), img, cor)

    # Cima.
    if r.c >= 0 and r.c < largura:
        img = desenha_linha(Coordenada(r.e, r.c), Coordenada(r.d, r.c), img, cor)

    # Baixo.
    if r.b >= 0 and r.b < largura:
        img = desenha_linha(Coordenada(r.e, r.b), Coordenada(r.d, r.b), img, cor)

    return img


def binariza(img, threshold):
    # TODO: implementar

    return img


def rotula(img, largura_min, altura_min, n_pixels_min):
    # TODO: implementar, mock de teste
    time.sleep(2)

    r1 = Retangulo(10, 100, 50, 200)
    r2 = Retangulo(300, 250, 450, 350)
    r3 = Retangulo(500, 550, 700, 750)

    return 3, [Componente(r1), Componente(r2), Componente(r3)]


if __name__ == "__main__":
    # .astype(np.float) é para abrir os valores como float e não int
    # Dessa forma, é possível normalizar em seguida
    img = cv2.imread(INPUT_IMAGE).astype(np.float)
    # Normalizando os valores dos pixels para ficar no intervalo [0, 1]
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # img.shape contem altura, largura e canais em uma tupla: (altura, largura, canais)
    # A função np.zeros cria uma matriz preenchida com zeros no formato passado
    # Ou seja, [altura][largura][canais]
    img_out = np.zeros(img.shape)
    for i in range(0, 3):
        # Aqui é um syntactic sugar do Python, ao invés de iterar na matriz
        # é possível copiar os valores usando array slicing
        # Ref: https://stackoverflow.com/questions/509211/understanding-slice-notation
        # Sabendo que a matriz "img" tem três dimensões, e acessamos um pixel assim:
        #
        # pixel = img[y][x][canal]
        #
        # A construção abaixo é equivalente a:
        #
        # for y in range(0, altura):
        #   for x in range(0, largura):
        #       for canal in range(0, canais):
        #           img_out[y][x][canal] = img[y][x][0]
        #
        # Ou seja, estamos copiando apenas o valor do canal 0 nos 3 canais de img_out!
        # O professor executa isso na função cinzaParaRgb no código em C:
        #
        # int i, j, k;
        # for (i = 0; i < 3; i++)
        #   for (j = 0; j < in->altura; j++)
        #        for (k = 0; k < in->largura; k++)
        #            out->dados [i][j][k] = in->dados [0][j][k];
        img_out[:, :, i] = img[:, :, 0]

    if NEGATIVO:
        img = inverte(img)

    img = binariza(img, THRESHOLD)
    salvar_imagem("01 - binarizada.bmp", img)

    tempo_inicio = datetime.now()
    # Aqui a função rotula() deve retornar dois elementos, em Python pode!
    n_componentes, componentes = rotula(img_out, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    tempo_total = datetime.now() - tempo_inicio

    print(f"Tempo: {tempo_total}")
    print(f"Componentes detectados: {n_componentes}")

    # Mostra os objetos encontrados
    for i in range(0, n_componentes):
        img_out = desenha_retangulo(componentes[i].retangulo, img_out)

    cv2.imwrite("02 - out.bmp", img_out)
