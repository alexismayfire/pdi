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

VERMELHO = np.array([0, 0, 1])


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


def desenha_retangulo(retangulo, img_out, cor=VERMELHO):
    return img_out


def binariza(img, threshold):
    return img


def rotula(img, largura_min, altura_min, n_pixels_min):
    time.sleep(2)

    return 10, None


if __name__ == "__main__":
    img = cv2.imread(INPUT_IMAGE).astype(np.float)
    # Normalizando os valores dos pixels para ficar entre 0 e 1
    # img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # img.shape contem altura, largura e canais em uma tupla
    img_out = np.zeros(img.shape)

    if NEGATIVO:
        img = inverte(img)

    img = binariza(img, THRESHOLD)
    cv2.imwrite("01 - binarizada.bmp", img)

    tempo_inicio = datetime.now()
    n_componentes, componentes = rotula(img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    tempo_total = datetime.now() - tempo_inicio

    print(f"Tempo: {tempo_total}")
    print(f"Componentes detectados: {n_componentes}")

    # Mostra os objetos encontrados
    for i in range(0, n_componentes):
        img_out = desenha_retangulo(componentes[i].retangulo, img_out)

    cv2.imwrite("02 - out.bmp", img_out)
