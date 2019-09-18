from colorsys import rgb_to_hls
from datetime import datetime
from multiprocessing import Pool, Process
from sys import float_info

import cv2
import numpy as np


# INPUT_NAME = "Wind Waker GC menor"
INPUT_NAME = "GT2"
INPUT_FORMAT = ".BMP"
THRESHOLD = 0.5


def salvar_imagem(nome, img):
    canais = None

    try:
        altura, largura, canais = img.shape
    except ValueError:
        # Imagem não tem 3 canais!
        altura, largura = img.shape

    saida = np.zeros(img.shape)

    for y in range(0, altura):
        for x in range(0, largura):
            if canais:
                for canal in range(0, canais):
                    saida[y][x][canal] = img[y][x][canal] * 255
            else:
                saida[y][x] = img[y][x] * 255

    cv2.imwrite(nome, saida)


# Converte uma imagem de rgb para hsl
def rgb_para_hsl(img):
    try:
        altura, largura, canais = img.shape
    except ValueError:
        raise ValueError("As imagens precisam ter 3 canais!")

    saida = np.zeros(img.shape)

    for y in range(0, altura):
        for x in range(0, largura):
            blue, green, red = img[y][x]
            vmax = max(red, max(green, blue))
            vmin = min(red, min(green, blue))
            croma = vmax - vmin

            # L
            l = (vmax + vmin) * 0.5

            # Trata de casos excepcionais
            if croma < float_info.epsilon:
                h = 0
                s = 0
            else:
                # S
                if l < 0.5:
                    s = croma / (vmax + vmin)
                else:
                    s = croma / (2 - vmax - vmin)

                if vmax == red:
                    h = 60 * (green - blue) / croma
                elif vmax == green:
                    h = 120 + 60 * (blue - red) / croma
                else:
                    h = 250 + 60 * (red - green) - croma

                # Para evitar problemas de arredondamento
                if h < 0:
                    h += 360.0

            saida[y][x][0] = h
            saida[y][x][1] = s
            saida[y][x][2] = l

    return saida


# Borra a imagem com o box blur
def box_blur(img, shape=3):
    if shape % 2 == 0:
        raise ValueError("O shape da janela precisa ser ímpar!")

    h = np.zeros(img.shape)
    v = np.zeros(img.shape)
    altura, largura, canais = img.shape
    janela = shape // 2

    for y in range(janela, altura - janela):
        for x in range(janela, largura - janela):
            # blue
            h[y][x][0] = sum(img[y, x - janela : x + janela + 1, 0]) / shape
            # green
            h[y][x][1] = sum(img[y, x - janela : x + janela + 1, 1]) / shape
            # red
            h[y][x][2] = sum(img[y, x - janela : x + janela + 1, 2]) / shape

    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            v[y][x][0] = sum(h[y - janela : y + janela + 1, x, 0]) / shape
            v[y][x][1] = sum(h[y - janela : y + janela + 1, x, 1]) / shape
            v[y][x][2] = sum(h[y - janela : y + janela + 1, x, 2]) / shape

    return v


def mask_blur(mask, op_number, shape=19, repeats=3):
    tempo_inicio = datetime.now()

    # Realiza o Filtro da Média de acordo com as repetições definidas
    for i in range(0, repeats):
        # print(f"OP number {op_number}, repeat: {i + 1}")
        mask = box_blur(mask, shape)

    tempo_total = datetime.now() - tempo_inicio
    print(f"Tempo: {tempo_total}")
    salvar_imagem(f"{INPUT_NAME}-mask-{op_number}.jpg", mask)

    return mask


def bloom(img, usar_mascara=False):

    # Converte a imagem de RGB para HSL
    img_hsl = rgb_para_hsl(img)

    # Primeira Mascara
    mascara = np.zeros(img.shape)

    # Passa os pixels com valores de L maiores que o THRESHOLD
    altura, largura, canais = img.shape
    for y in range(0, altura):
        for x in range(0, largura):
            h, s, l = img_hsl[y][x]
            if l > THRESHOLD:
                mascara[y][x] = img[y][x]
            else:
                mascara[y][x] = np.array([0.0, 0.0, 0.0])

    """
    args = [
        (mascara, 1, 3),
        (mascara, 2, 6),
        (mascara, 3, 9),
        (mascara, 4, 12)
    ]

    pool = Pool(processes=4)
    data = pool.map(mask_blur, args)
    pool.close()
    blur_1, blur_2, blur_3, blur_4 = data
    """

    if not usar_mascara:
        # Cria as imagens cada vez mais borradas a partir da original
        blur_1 = mask_blur(mascara, 1, repeats=5)
        blur_2 = mask_blur(blur_1, 2, repeats=5)
        blur_3 = mask_blur(blur_2, 3, repeats=5)
        blur_4 = mask_blur(blur_3, 4, repeats=5)

        # Soma todas as imagens borradas limitando o valor de brilho a 5
        mascara_borrada = np.zeros(img.shape)
        for y in range(0, altura):
            for x in range(0, largura):
                for canal in range(0, canais):
                    valor = (
                        blur_1[y][x][canal]
                        + blur_2[y][x][canal]
                        + blur_3[y][x][canal]
                        + blur_4[y][x][canal]
                    )
                    mascara_borrada[y][x][canal] = valor if valor <= 5.0 else 5.0

    else:
        mascara_borrada = cv2.imread(f"{INPUT_NAME}-mask-final.jpg").astype(np.float)
        mascara_borrada = cv2.normalize(
            mascara_borrada, None, 0.0, 1.0, cv2.NORM_MINMAX
        )

    # Salva a máscara final para visualização
    salvar_imagem(f"{INPUT_NAME}-mask-final.jpg", mascara_borrada)

    saida = np.zeros(img.shape)
    alfa = 0.8
    beta = 0.45
    brilho = -0.1
    contraste = 1.2

    for y in range(0, altura):
        for x in range(0, largura):
            for canal in range(0, canais):
                saida[y][x][canal] = (
                    (img[y][x][canal] * alfa)
                    + (mascara_borrada[y][x][canal] * beta)
                    + 0.5
                    + brilho
                )
                saida[y][x][canal] = (saida[y][x][canal] - 0.5) * contraste

    return saida


if __name__ == "__main__":
    img = cv2.imread(f"{INPUT_NAME}{INPUT_FORMAT}").astype(np.float)
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    tempo_inicio = datetime.now()

    # Efeito de Bloom
    img_bloom = bloom(img)

    tempo_total = datetime.now() - tempo_inicio
    print(f"Tempo: {tempo_total}")
    salvar_imagem(f"{INPUT_NAME}.jpg", img_bloom)
