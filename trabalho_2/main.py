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


def soma_pixel(pixel, valor_janela):
    blue, green, red = pixel
    valor_janela["blue"] += blue
    valor_janela["green"] += green
    valor_janela["red"] += red

    return valor_janela


class Ingenuo:
    @staticmethod
    def janela(shape, img, y, x):

        # Passa os 3 canais do pixel para variáveis separadas
        blue, green, red = img[y][x]
        # Cria um dicionário para os canais
        valor_janela = {"blue": np.float(0), "green": np.float(0), "red": np.float(0)}
        # Define um limite para percorrer a janela
        limit = (shape // 2) + 1

        # Percorre a janela
        # range(1, 3)
        # só vai iterar usando o 1!
        for y_start in range(1, limit):
            # Linha de cima, mesma coluna
            valor_janela = soma_pixel(img[y - y_start][x], valor_janela)
            # Linha de baixo, mesma coluna
            valor_janela = soma_pixel(img[y + y_start][x], valor_janela)
            for x_start in range(1, limit):
                # Linha de cima, coluna da esquerda
                valor_janela = soma_pixel(img[y - y_start][x - x_start], valor_janela)
                # Linha de cima, coluna da direita
                valor_janela = soma_pixel(img[y - y_start][x + x_start], valor_janela)
                # Linha corrente, coluna da esquerda
                valor_janela = soma_pixel(img[y][x - x_start], valor_janela)
                # Linha corrente, coluna da direita
                valor_janela = soma_pixel(img[y][x + x_start], valor_janela)
                # Linha de baixo, coluna da esquerda
                valor_janela = soma_pixel(img[y + y_start][x - x_start], valor_janela)
                # Linha de baixo, coluna da direita
                valor_janela = soma_pixel(img[y + y_start][x + x_start], valor_janela)

        valor_pixel = []
        for canal, valor in valor_janela.items():
            valor_pixel.append(valor / shape ** 2)

        return np.asarray(valor_pixel)

    @staticmethod
    def blur(img, shape=3):
        if shape % 2 == 0:
            raise ValueError("O shape da janela precisa ser ímpar!")

        new_img = np.zeros(img.shape)
        altura, largura, canais = img.shape

        for y in range(1, altura - shape // 2):
            for x in range(1, largura - shape // 2):
                pixel = Ingenuo.janela(shape, img, y, x)
                new_img[y][x] = pixel

        return new_img


class Separavel:
    @staticmethod
    def blur(img, shape=3):
        if shape % 2 == 0:
            raise ValueError("O shape da janela precisa ser ímpar!")

        new_img = np.zeros(img.shape)
        altura, largura, canais = img.shape

        for y in range(1, altura - shape // 2):
            for x in range(1, largura - shape // 2):
                new_img[y][x] = img[y][x - 1] + img[y][x] + img[y][x + 1]


if __name__ == "__main__":
    img = cv2.imread(INPUT_IMAGE).astype(np.float)
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    img_blur = Ingenuo.blur(img)

    salvar_imagem("com blur.jpg", img_blur)
