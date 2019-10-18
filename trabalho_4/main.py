from math import sqrt

import cv2
import numpy as np
from matplotlib import pyplot as plt


IMAGES = ["60", "82", "114", "150", "205"]
EXTENSION = ".bmp"
COLORS = ["b", "m", "y", "k", "r"]


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


class Coordenada:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def inunda(label, img, y, x):
    altura, largura = img.shape
    img[y][x] = label

    # Analisa se existem pixels vizinhos, pois perto das margens não terão
    # Cima
    try:
        if y > 0 and img[y - 1][x] == -1:
            img = inunda(label, img, y - 1, x)
        # Direita
        if x < largura - 1 and img[y][x + 1] == -1:
            img = inunda(label, img, y, x + 1)
        # Baixo
        if y < altura - 1 and img[y + 1][x] == -1:
            img = inunda(label, img, y + 1, x)
        # Esquerda
        if x > 0 and img[y][x - 1] == -1:
            img = inunda(label, img, y, x - 1)
    except RecursionError:
        print(y, x)

    return img


def rotula(img, largura_min, altura_min, n_pixels_min):
    componentes = []
    label = 1

    altura, largura = img.shape

    for y in range(0, altura):
        for x in range(0, largura):
            if img[y][x] == -1:
                img = inunda(label, img, y, x)
                ret = Retangulo(9999, -1, 9999, -1)
                comp = Componente(ret)
                comp.label = label
                componentes.append(comp)
                label += 1

    for y in range(0, altura):
        for x in range(0, largura):
            # Agora, como atualizamos img (que antes possuía valores binários, [0, -1]) com os RÓTULOS
            # A matriz vai conter valores [0...n], onde n é a quantidade de rótulos identificados no passo anterior
            if img[y][x] > 0:
                # Para pegar o índice do componente
                # Como label começa em 1, precisa diminuir
                indice_comp = int(img[y][x]) - 1
                componentes[indice_comp].n_pixels += 1

                # Para atualizar o menor valor de Y para o label corrente (cima)
                if y < componentes[indice_comp].retangulo.c:
                    componentes[indice_comp].retangulo.c = y

                # Para atualizar o maior valor de Y para o label corrente (baixo)
                if y > componentes[indice_comp].retangulo.b:
                    componentes[indice_comp].retangulo.b = y

                # Para atualizar o menor valor de X para o label corrente (esquerda)
                if x < componentes[indice_comp].retangulo.e:
                    componentes[indice_comp].retangulo.e = x

                # Para atualizar o maior valor de X para o label corrente (direita)
                if x > componentes[indice_comp].retangulo.d:
                    componentes[indice_comp].retangulo.d = x

    # Agora, iteramos na lista de componentes para identificar os que são pequenos demais e remover
    aux = componentes
    componentes = []
    for componente in aux:
        altura_componente = componente.retangulo.b - componente.retangulo.c
        largura_componente = componente.retangulo.d - componente.retangulo.e
        if (
            componente.n_pixels > n_pixels_min
            and altura_componente > altura_min
            and largura_componente > largura_min
        ):
            componentes.append(componente)

    return componentes


def contagem(componentes):
    soma = 0
    qtde = len(componentes)

    for comp in componentes:
        soma += comp.n_pixels

    # Tamanho médio de cada componente
    media = soma / len(componentes)

    for comp in componentes:
        if comp.n_pixels >= int(media * 1.5):
            qtde += 2
        elif comp.n_pixels >= int(media * 3.5):
            qtde += 3
        elif comp.n_pixels >= int(media * 5.5):
            qtde += 4

    return qtde


for image in IMAGES:
    img = cv2.imread(f"{image}{EXTENSION}").astype(np.float32)
    # img = cv2.normalize(img, None, np.float(0), np.float(1), cv2.NORM_MINMAX)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape
    grey[:, :] = grey[:, :] / 255

    final = np.zeros(grey.shape, dtype=np.uint8)
    gradiente = np.zeros(grey.shape)
    img_rotulada = np.zeros(grey.shape)

    grey = cv2.GaussianBlur(grey, (9, 9), 1)
    sobelx = np.absolute(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
    sobely = np.absolute(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
    gradiente[:, :] = sobelx[:, :] + sobely[:, :]

    aux = np.zeros(grey.shape)
    aux[:, :] = gradiente[:, :] + grey[:, :]

    for y in range(0, height):
        for x in range(0, width):
            if aux[y][x] > 1.1:
                final[y, x] = 255
            else:
                final[y, x] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)

    for y in range(0, height):
        for x in range(0, width):
            if final[y, x] == 255:
                img_rotulada[y, x] = -1

    cv2.imwrite(f"{image} bin.png", final)

    comp = rotula(img_rotulada, 10, 10, 10)
    print(contagem(comp))

    # cv2.imshow(f"Gradiente {image}", final)
    # cv2.waitKey(0)

cv2.destroyAllWindows()

