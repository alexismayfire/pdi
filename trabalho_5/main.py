from statistics import mean
from os import path

import cv2
import numpy as np
from matplotlib import pyplot as plt

INPUT = "/img"
OUTPUT = "/res"

"""
Estratégia, do grego strategia, do inglês strategy

1. Dividir a imagem em canais
2. Ver como fica o canal verde x vermelho e azul
    (X) Binarizando cada canal separadamente com Otsu;
        Salvando o valor de threshold em um acumulador;
        Usando como média no canal no inRange para os lower/upper bounds

    (X) Usando gradiente para detectar bordas:
        Como melhorar? Gradiente só no canal verde não funciona

    

Tutorial: https://medium.com/fnplus/blue-or-green-screen-effect-with-open-cv-chroma-keying-94d4a6ab2743
"""

for i in range(0, 9):
    # print(f"Imagem {i}.bmp")
    img = cv2.imread(f"img/{i}.bmp").astype(np.uint8)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    blue, green, red = cv2.split(img)

    # Aplica Otsu binarizando cada canal de cor média ret = o limiar, th = imagem binarizada
    ret_b, th_blue = cv2.threshold(blue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret_g, th_green = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret_r, th_red = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lower_green = np.array([0, ret_g, 0])
    upper_green = np.array([140.0, 255.0, 170.0])

    mask = cv2.inRange(img, lower_green, upper_green)
    plt.imshow(mask, cmap="gray")
    plt.show()
