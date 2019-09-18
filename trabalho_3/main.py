from argparse import ArgumentParser, RawTextHelpFormatter
from colorsys import rgb_to_hls
from datetime import datetime
from sys import argv, float_info

import cv2
import numpy as np


INPUTS = ["GT2", "GT2 menor", "Wind Waker GC", "Wind Waker GC menor"]
INPUT_NAME = "Wind Waker GC"
INPUT_FORMAT = ".BMP"
THRESHOLD = 0.5
ALFA = 1.1
BETA = 0.2
BRILHO = -0.1
CONTRASTE = 1.1

TIPO_BLUR = None
GAUSSIAN_BLUR = "gaussian"
BOX_BLUR = "box"
BOX_BLUR_WINDOW = 19
BOX_BLUR_REPEATS = 3


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
    # Transcrevemos a função em C disponibilizada pelo professor
    
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


def box_blur_integral(img, shape=BOX_BLUR_WINDOW):
    # Transcrevemos a função em C disponibilizada pelo professor

    if shape % 2 == 0:
        raise ValueError("O shape da janela precisa ser ímpar")

    img_integral = np.zeros(img.shape)
    saida = np.zeros(img.shape)
    altura, largura, canais = img.shape

    for canal in range(0, canais):
        # Soma em linhas
        for y in range(0, altura):
            img_integral[y][0][canal] = img[y][0][canal]

            for x in range(1, largura):
                img_integral[y][x][canal] = (
                    img[y][x][canal] + img_integral[y][x - 1][canal]
                )

        # Agora soma na vertical
        for y in range(1, altura):
            for x in range(0, largura):
                img_integral[y][x][canal] += img_integral[y - 1][x][canal]

    # Agora calcula as médias
    for canal in range(0, canais):
        for y in range(0, altura):
            for x in range(0, largura):
                top = max(-1, y - shape // 2 - 1)
                left = max(-1, x - shape // 2 - 1)
                bottom = min(altura - 1, y + shape // 2)
                right = min(largura - 1, x + shape // 2)

                aux1, aux2, aux3 = [0, 0, 0]
                if top >= 0 and left >= 0:
                    aux1 = img_integral[top][left][canal]

                if left >= 0:
                    aux2 = img_integral[bottom][left][canal]

                if top >= 0:
                    aux3 = img_integral[top][right][canal]

                soma = aux1 + img_integral[bottom][right][canal] - aux2 - aux3
                area = (right - left) * (bottom - top)

                saida[y][x][canal] = soma / area

    return saida


def gaussian_blur(mask, op_number, nome_imagem, sigma=0):
    # Realiza o gaussian blur de acordo com o sigma
    # Usamos o padrão do CV2, já que no pacote de implementação em C,
    # o professor disponibilizou a função pronta para os alunos!
    mask = cv2.GaussianBlur(mask, (BOX_BLUR_WINDOW, BOX_BLUR_WINDOW), sigma)

    # Salvamos as etapas de cada aplicação do blur para comparar
    salvar_imagem(f"{nome_imagem}-mask-{TIPO_BLUR}-{op_number}.jpg", mask)

    return mask


def mask_box_blur(mask, op_number, nome_imagem):
    # Realiza o Filtro da Média de acordo com as repetições pré-definidas
    for i in range(0, BOX_BLUR_REPEATS):
        mask = box_blur_integral(mask)

    # Salvamos as etapas de cada aplicação do blur para comparar
    salvar_imagem(f"{nome_imagem}-mask-{TIPO_BLUR}-{op_number}.jpg", mask)

    return mask


def bloom(img, nome_imagem, usar_mascara=False):

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

    if not usar_mascara:
        # Cria as imagens cada vez mais borradas a partir da original
        if TIPO_BLUR == GAUSSIAN_BLUR:
            blur_1 = gaussian_blur(mascara, 1, nome_imagem, sigma=10)
            blur_2 = gaussian_blur(blur_1, 2, nome_imagem, sigma=10)
            blur_3 = gaussian_blur(blur_2, 3, nome_imagem, sigma=10)
            blur_4 = gaussian_blur(blur_3, 4, nome_imagem, sigma=10)
        else:
            blur_1 = mask_box_blur(mascara, 1, nome_imagem)
            blur_2 = mask_box_blur(blur_1, 2, nome_imagem)
            blur_3 = mask_box_blur(blur_2, 3, nome_imagem)
            blur_4 = mask_box_blur(blur_3, 4, nome_imagem)

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
                    # Limitamos o valor final a 5.0, para não estourar demais!
                    mascara_borrada[y][x][canal] = valor if valor <= 5.0 else 5.0

    else:
        # Quando usamos os mesmos parâmetros, só carregamos a máscara
        # A ideia é economizar tempo quando queremos apenas mexer nos parâmetros
        # de combinação da imagem original com a máscara!
        mascara_borrada = cv2.imread(
            f"{nome_imagem}-mask-final-{TIPO_BLUR}.jpg"
        ).astype(np.float)
        mascara_borrada = cv2.normalize(
            mascara_borrada, None, 0.0, 1.0, cv2.NORM_MINMAX
        )

    # Salva a máscara final para visualização, de acordo com o tipo de blur
    salvar_imagem(f"{nome_imagem}-mask-final-{TIPO_BLUR}.jpg", mascara_borrada)

    saida = np.zeros(img.shape)

    for y in range(0, altura):
        for x in range(0, largura):
            for canal in range(0, canais):
                # Ajustes da "mistura" da imagem, com valores pré-definidas, 
                # mas que podem ser setados pela linha de comando
                # Também aplicamos ajuste de brilho e contraste
                saida[y][x][canal] = (
                    (img[y][x][canal] * ALFA)
                    + (mascara_borrada[y][x][canal] * BETA)
                    + 0.5
                    + BRILHO
                )
                saida[y][x][canal] = (saida[y][x][canal] - 0.5) * CONTRASTE

    return saida


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Parses command", formatter_class=RawTextHelpFormatter
    )
    help_input = (
        "Selecione a imagem desejada: \n"
        "1 - GT2.bmp\n"
        "2 - GT2 menor.bmp\n"
        "3 - Wind Waker GC.bmp\n"
        "4 - Wind Waker GC menor.bmp\n"
    )
    parser.add_argument(
        "-i", 
        "--input", 
        dest="imagem", 
        type=int, 
        required=True, 
        help=help_input
    )

    help_carregar_mascara = (
        "Caso queira apenas alterar os parâmetros de combinação da imagem final.\n"
        "Será carregada uma máscara existente de acordo com o tipo de blur desejado."
    )
    parser.add_argument(
        "--carregar-mascara", 
        dest="carregar_mascara", 
        action="store_true", 
        help=help_carregar_mascara
    )
    # Por padrão, vamos sempre criar a máscara se não foi passado o comando
    parser.set_defaults(carregar_mascara=False)

    # Configurações adicionais da "mistura" da imagem
    parser.add_argument("--alfa", dest="alfa", type=float)
    parser.add_argument("--beta", dest="beta", type=float)
    parser.add_argument("--brilho", dest="brilho", type=float)
    parser.add_argument("--contraste", dest="contraste", type=float)

    # Para usar gaussian blur
    help_gaussian_blur = (
        "Parâmetro usado para borrar a máscara usando gaussian blur.\n" 
        "Caso seja omitido, será usado box blur"
    )
    parser.add_argument(
        "--gaussian-blur", 
        dest="gaussian_blur", 
        action="store_true", 
        help=help_gaussian_blur
    )
    # Por padrão, usa o box blur
    parser.set_defaults(gaussian_blur=False)

    options = parser.parse_args(argv[1:])

    nome_imagem = INPUTS[options.imagem - 1]
    # Redefinindo os valores caso tenham sido passados pela linha de comando
    if options.alfa:
        ALFA = options.alfa
    if options.beta:
        BETA = options.beta
    if options.brilho:
        BRILHO = options.brilho
    if options.contraste:
        CONTRASTE = options.contraste
    TIPO_BLUR = GAUSSIAN_BLUR if options.gaussian_blur else BOX_BLUR

    img = cv2.imread(f"{nome_imagem}{INPUT_FORMAT}").astype(np.float)
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    tempo_inicio = datetime.now()

    # Efeito de Bloom
    # Se usamos a opção para carregar uma máscara, apenas vamos tratar a 
    # combinação da máscara com a imagem original!
    img_bloom = bloom(img, nome_imagem, usar_mascara=options.carregar_mascara)

    tempo_total = datetime.now() - tempo_inicio
    print(f"Tempo: {tempo_total}")
    # Salvamos as imagens com os parâmetros usados para comparação!
    nome_saida = (
        f"{nome_imagem}-"
        f"alfa={ALFA}-"
        f"beta={BETA}-"
        f"brilho={BRILHO}-"
        f"contraste={CONTRASTE}-"
        f"blur={TIPO_BLUR}.jpg"
    )
    salvar_imagem(nome_saida, img_bloom)

