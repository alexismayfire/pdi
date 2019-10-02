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

    return img


def flood_fill(img, pilha, comp):
    n_pilha = 1
    altura, largura = img.shape
    img[pilha[0].y, pilha[0].x] = comp.label

    while n_pilha > 0:
        c = pilha[n_pilha - 1]
        n_pilha -= 1
        comp.n_pixels += 1

        if c.y < comp.retangulo.c:
            comp.retangulo.c = c.y
        if c.y > comp.retangulo.b:
            comp.retangulo.b = c.y
        if c.x < comp.retangulo.e:
            comp.retangulo.e = c.x
        if c.x > comp.retangulo.d:
            comp.retangulo.d = c.x

        if c.x > 0 and img[c.y, c.x - 1] < 0:
            img[c.y, c.x - 1] = comp.label
            pilha.append(Coordenada(c.x - 1, c.y))
            n_pilha += 1
        if c.x < largura - 1 and img[c.y, c.x + 1] < 0:
            img[c.y, c.x + 1] = comp.label
            pilha.append(Coordenada(c.x + 1, c.y))
            n_pilha += 1
        if c.y > 0 and img[c.y - 1, c.x] < 0:
            img[c.y - 1, c.x] = comp.label
            pilha.append(Coordenada(c.x, c.y - 1))
            n_pilha += 1
        if c.y < altura - 1 and img[c.y + 1, c.x] < 0:
            img[c.y + 1, c.x] = comp.label
            pilha.append(Coordenada(c.x, c.y + 1))
            n_pilha += 1


def rotula_2(img, largura_min, altura_min, n_pixels_min):
    componentes = []
    pilha = []
    pilha.append(None)
    n = 0
    label = 0.1
    altura, largura = img.shape

    for y in range(0, altura):
        for x in range(0, largura):
            if img[y, x] > 0:
                img[y, x] = -1

    for y in range(0, altura):
        for x in range(0, largura):
            if img[y, x] < 0:
                ret = Retangulo(y, y, x, x)
                comp = Componente(ret)
                comp.label = label
                comp.n_pixels = 0

                pilha[0] = Coordenada(x, y)
                flood_fill(img, pilha, comp)

                if (
                    comp.n_pixels >= n_pixels_min
                    and comp.retangulo.d - comp.retangulo.e + 1 >= largura_min
                    and comp.retangulo.b - comp.retangulo.c + 1 >= altura_min
                ):
                    n += 1
                label += 0.1

    return (componentes, n)


def inunda(label, img, y, x):
    altura, largura = img.shape
    img[y][x] = label

    # Analisa se existem pixels vizinhos, pois perto das margens não terão
    # Cima
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

    return img


def rotula(img, largura_min, altura_min, n_pixels_min):
    componentes = []
    label = 1

    altura, largura = img.shape

    # Trocando tudo que foi marcado como branco na binariza() para -1 (foreground)
    # Esse passo talvez não fosse necessário. Na binariza(), poderia marcar direto como -1!
    # Caso quisesse salvar a imagem binarizada apenas, teria que alterar a salvar_imagem(),
    # para multiplicar por 'abs(img[y][x]) * 255'.
    #
    # Outra solução: ao binarizar a imagem, criar UM canal.
    # Ou seja, os valores dos pixels seriam acessados como img[y][x][0].
    # Na identificação de rótulos, ao invés de substituir o valor -1 (foreground) em img,
    # poderia colocar no "segundo canal": img[y][x][1] = label
    # Ao final dessa função, removemos essa posição extra (ou não, porque img não é mais usada na main) 

    for y in range(0, altura):
        for x in range(0, largura):
            if img[y][x] == -1:
                img = inunda(label, img, y, x)
                # Inicializando o retângulo com valores muito altos ou muito pequenos,
                # no próximo passo identificamos as coordenadas limite de acordo com o label
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



def plot_histograms():
    plt.subplot(2, 1, 1)
    for image, color in zip(IMAGES, COLORS):
        img = cv2.imread(f"{image}{EXTENSION}")
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([img], [0], None, [16], [0, 255])
        plt.plot(hist, color, label=image)
        plt.xlim([0, 16])

    plt.title("Comparação histograma")
    plt.legend(loc="best")
    plt.ylabel("Normalizadas")

    plt.subplot(2, 1, 2)
    for image, color in zip(IMAGES, COLORS):
        img = cv2.imread(f"{image}{EXTENSION}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([img], [0], None, [16], [0, 255])
        plt.plot(hist, color, label=image)
        plt.xlim([0, 16])

    plt.legend(loc="best")
    plt.xlabel("Valor")
    plt.ylabel("Originais")

    plt.show()


def show_gray(normalize=False, as_float=False, save=False):
    for image in IMAGES:
        if as_float:
            img = cv2.imread(f"{image}{EXTENSION}").astype(np.float32)
            if normalize:
                img = cv2.normalize(
                    img, None, np.float(0), np.float(1), cv2.NORM_MINMAX
                )
        else:
            img = cv2.imread(f"{image}{EXTENSION}")
            if normalize:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        """ 
        grey = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        scale = 60
        width = int(img.shape[1] * scale / 100)
        height = int(img.shape[0] * scale / 100)
        img = cv2.resize(img, (width, height))
        grey = cv2.resize(grey, (width, height))
        concatenadas = np.concatenate((img, grey), axis=1)
        cv2.imshow("Original x Escala de Cinza", concatenadas)
        cv2.waitKey(0)
        """

        if save:
            if as_float and normalize:
                grey[:, :] = grey[:, :] * 255
            cv2.imwrite(f"{image}-grey-norm={normalize}-float={as_float}.png", grey)

    # cv2.destroyAllWindows()


def preprocessing():
    for image, color in zip(IMAGES, COLORS):
        img = cv2.imread(f"{image}{EXTENSION}").astype(np.float32)
        # img = cv2.normalize(img, None, np.float(0), np.float(1), cv2.NORM_MINMAX)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey[:, :] = grey[:, :] / 255
        height, width = grey.shape

        """
        final = np.zeros(grey.shape)
        gradiente = np.zeros(grey.shape)

        grey = cv2.GaussianBlur(grey, (5, 5), 1)
        sobelx = np.absolute(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
        sobely = np.absolute(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
        gradiente[:, :] = sobelx[:, :] + sobely[:, :]

        print(gradiente.max(), gradiente.min())
        print(grey.max(), grey.min())

        aux = np.zeros(grey.shape)
        aux[:, :] = gradiente[:, :] + grey[:, :]
        print(aux.max(), aux.min())

        for y in range(0, height):
            for x in range(0, width):
                if aux[y][x] > 1:
                    final[y, x] = 255
                else:
                    final[y, x] = 0
        # aux = cv2.normalize(aux, None, np.float(0), np.float(1), cv2.NORM_MINMAX)
        # final[:, :] = aux[:, :] * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
        """

        integral = np.zeros(grey.shape, np.float32)
        # Soma em linhas.
        for row in range(0, height):
            integral[row, 0] = grey[row, 0]

            for col in range(1, width):
                integral[row, col] = grey[row, col] + integral[row, col - 1]

        # Agora soma na vertical.
        for row in range(1, height):
            for col in range(0, width):
                integral[row, col] += integral[row - 1, col]

        box_y = 25
        box_x = 25

        medias = np.zeros(grey.shape, np.float32)
        final = np.zeros(grey.shape, np.uint8)

        for row in range(0, height):
            for col in range(0, width):
                top = max(-1, row - box_y // 2 - 1)
                left = max(-1, col - box_x // 2 - 1)
                bottom = min(height - 1, row + box_y // 2)
                right = min(width - 1, col + box_x // 2)

                a = integral[top, left] if (top >= 0 and left >= 0) else 0
                b = integral[top, right] if top >= 0 else 0
                c = integral[bottom, left] if left >= 0 else 0
                d = integral[bottom, right]

                soma = a + d - b - c

                area = (right - left) * (bottom - top)
                medias[row, col] = soma / area

        # medias = cv2.GaussianBlur(medias, (13, 13), 1)
        hist = cv2.calcHist([medias], [0], None, [10], [0.0, 1.0])
        maior = 0
        index = 0
        for i in range(0, len(hist)):
            if hist[i] > maior:
                maior = hist[i]
                index = i

        threshold = index / len(hist)
        print(threshold)

        for y in range(0, height):
            for x in range(0, width):
                if (grey[y, x] - medias[y, x]) > 0.1:
                    final[y, x] = 255
                else:
                    final[y, x] = 0
        """
        # aux = np.zeros(grey.shape, dtype=np.uint8)
        # aux[:, :] = grey[:, :] * 255

        # grey = cv2.equalizeHist(aux)
        # grey = cv2.GaussianBlur(grey, (5, 5), 0)

        # thresh = np.zeros(grey.shape, dtype=np.uint8)
        # thresh[:, :] = np.uint8(grey[:, :] * 255)

        # hist = cv2.calcHist([thresh], [0], None, [256], [0, 255])
        # print(image, len(hist))
        # print("-----------------------")
        # plt.plot(hist, color, label=image)
        # plt.xlim([0, 256])

        # final = cv2.adaptiveThreshold(
        #     aux, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 0.2
        # )
        """

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final = cv2.morphologyEx(final, cv2.MORPH_OPEN, morph_kernel)
        final_rotulagem = np.zeros(final.shape)
        for y in range(0, height):
            for x in range(0, width):
                if final[y, x] == 255:
                    final_rotulagem = -1
                else:
                    final_rotulagem = 0

        comp = rotula(final, 5, 5, 20)
        print(len(comp))

        print("-----")

        cv2.imshow(f"Gradiente {image}", final)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # plt.legend(loc="best")
    # plt.xlabel("Valor")
    # plt.ylabel("Originais")

    # plt.show()


preprocessing()

