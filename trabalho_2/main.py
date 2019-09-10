from datetime import datetime

import cv2
import numpy as np


INPUT_IMAGE = "imagem.jpg"


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


class Ingenuo:
    @staticmethod
    def blur(img, shape=3):
        if shape % 2 == 0:
            raise ValueError("O shape da janela precisa ser ímpar!")

        new_img = np.zeros(img.shape)
        altura, largura, canais = img.shape
        janela = shape // 2

        for y in range(janela, altura - janela):
            for x in range(janela, largura - janela):
                y1 = y - janela
                y2 = y + janela + 1
                x1 = x - janela
                x2 = x + janela + 1
                new_img[y][x][0] = sum(sum(img[y1:y2, x1:x2, 0])) / (shape ** 2)
                new_img[y][x][1] = sum(sum(img[y1:y2, x1:x2, 1])) / (shape ** 2)
                new_img[y][x][2] = sum(sum(img[y1:y2, x1:x2, 2])) / (shape ** 2)

        return new_img


class Separavel:
    @staticmethod
    def blur(img, shape=3):
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


if __name__ == "__main__":
    img = cv2.imread(INPUT_IMAGE).astype(np.float)
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    tempo_inicio = datetime.now()
    img_blur = Ingenuo.blur(img, shape=11)
    tempo_total = datetime.now() - tempo_inicio
    print(f"Tempo: {tempo_total}")
    salvar_imagem("com blur.jpg", img_blur)

    tempo_inicio = datetime.now()
    img_blur = Separavel.blur(img, shape=11)
    tempo_total = datetime.now() - tempo_inicio
    print(f"Tempo: {tempo_total}")
    salvar_imagem("com blur separado.jpg", img_blur)
