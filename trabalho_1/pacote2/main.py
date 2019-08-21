from datetime import datetime
import time

import cv2
import numpy as np

INPUT_IMAGE = "arroz.bmp"
NEGATIVO = 0
THRESHOLD = 0.4
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 1

# TODO: alterar para 1 quando abrir em escala de float
VERMELHO = np.array([0, 0, 255])


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


def inverte(img):
    img_invertida = np.zeros(img.shape)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            b, g, r = img[y][x]
            # Isso aqui funciona corretamente apenas se a imagem for aberta normalizada!
            img_invertida[y][x][0] = np.float(1) - b
            img_invertida[y][x][1] = np.float(1) - g
            img_invertida[y][x][2] = np.float(1) - r

    return img_invertida


def desenha_linha(p1, p2, img, cor):
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
    altura, largura, _ = img.shape
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
    img = cv2.imread(INPUT_IMAGE).astype(np.float)
    # Normalizando os valores dos pixels para ficar entre 0 e 1
    # img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # img.shape contem altura, largura e canais em uma tupla
    img_out = np.zeros(img.shape)
    for i in range(0, 3):
        img_out[:, :, i] = img[:, :, 0]

    if NEGATIVO:
        img = inverte(img)

    img = binariza(img, THRESHOLD)
    cv2.imwrite("01 - binarizada.bmp", img)

    tempo_inicio = datetime.now()
    n_componentes, componentes = rotula(img_out, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    tempo_total = datetime.now() - tempo_inicio

    print(f"Tempo: {tempo_total}")
    print(f"Componentes detectados: {n_componentes}")

    # Mostra os objetos encontrados
    for i in range(0, n_componentes):
        img_out = desenha_retangulo(componentes[i].retangulo, img_out)

    cv2.imwrite("02 - out.bmp", img_out)
