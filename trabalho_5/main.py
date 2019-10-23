from statistics import mean
from os import path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

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

    () Fazer uma cópia da imagem -> Borrar -> encontrar limiar com Otsu da imagem borrada
        Depois aplicar o algoritmo com os números mágicos

    () Aumentar o contraste da imagem para que o Sobel fique mais definido nas bordas
       Depois preencher o desenho dentro do sobel para destacar uma máscara bem feita

    () Usar o princípio de HDR, para cada imagem gerar outras N com variações de brilho e contraste
       Depois, criar o "mapa" usando a abordagem #2 dos slides (variação) 
3. Usar o espaço HSV

    

Tutorial: https://medium.com/fnplus/blue-or-green-screen-effect-with-open-cv-chroma-keying-94d4a6ab2743
"""


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(
            bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1
        )
        startX = endX

    # return the bar chart
    return bar


def show_color_palette(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))
    cluster = KMeans(10)
    cluster.fit(rgb)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(cluster)
    bar = plot_colors(hist, cluster.cluster_centers_)

    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()


class HsvColor:
    def __init__(self, rgb_code):
        self.hue = cv2.cvtColor(np.uint8([[rgb_code]]), cv2.COLOR_RGB2HSV)[0][0][0]
        self.upper_bound = self.hue + 29  # valores variam de 0-30 no OpenCV
        self.hue_mean = None
        self.hue_values = []
        self.sat_mean = None
        self.sat_values = []
        self.v_mean = None
        self.v_values = []

    def calculate_mean(self):
        try:
            self.hue_mean = sum(self.hue_values) / len(self.hue_values)
        except ZeroDivisionError:
            print(f"Nenhum pixel com o hue [{self.hue}-{self.upper_bound}]")
            return

        self.sat_mean = sum(self.sat_values) / len(self.sat_values)
        self.v_mean = sum(self.v_values) / len(self.v_values)


class HsvCluster:
    def __init__(self):
        self.red = HsvColor([255, 0, 0])  # 0-29
        self.yellow = HsvColor([255, 255, 0])  # 30-59
        self.green = HsvColor([0, 255, 0])  # 60-89
        self.cyan = HsvColor([0, 255, 255])  # 90-119
        self.blue = HsvColor([0, 0, 255])  # 120-149
        self.magenta = HsvColor([255, 0, 255])  # 150-179
        self.colors = [
            self.red,
            self.yellow,
            self.green,
            self.cyan,
            self.blue,
            self.magenta,
        ]

    def calculate_pixel(self, pixel):
        hue, sat, v = pixel

        for color in self.colors:
            if hue <= color.upper_bound:
                color.hue_values.append(hue)
                color.sat_values.append(sat)
                color.v_values.append(v)
                break

    def calculate_colors_mean(self):
        for color in self.colors:
            color.calculate_mean()


def show_cluster_values(cluster):
    colors = ["red", "yellow", "green", "cyan", "blue", "magenta"]
    hue_ranges = ["[0-29]", "[30-59]", "[60-89]", "[90-119]", "[120-149]", "[150-179]"]
    for color, hue_range in zip(colors, hue_ranges):
        c = getattr(cluster, color)
        print(f"{color} {hue_range}\t: {c.hue_mean} \t| {c.sat_mean} \t| {c.v_mean}")


def simple_mask(img, background, hsv, lower_green, upper_green):
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_3d = np.zeros(img.shape, dtype=np.uint8)
    for ch in range(img.shape[2]):
        mask_3d[:, :, ch] = mask[:, :]

    output = np.zeros(img.shape, dtype=np.uint8)
    output[:, :, :] = np.where(mask_3d == 0, img, background)

    return output


def weighted_mask(img, background, hsv, lower_green, upper_green):
    height, width, channels = img.shape
    mask = np.full((height, width), 1, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            valid = True
            for ch in range(channels):
                if not (
                    hsv[y, x, ch] >= lower_green[ch]
                    and hsv[y, x, ch] <= upper_green[ch]
                ):
                    valid = False
                    break
            if valid:
                mask[y, x] = 0

    print("Calculando os pesos...")
    alpha_mask = np.zeros((height, width), dtype=np.float32)
    alpha_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    bg_weights = np.zeros((height, width), dtype=np.float32)
    bg_weights[:, :] = 1 - alpha_mask[:, :]
    img_weights = np.zeros((height, width), dtype=np.float32)
    img_weights[:, :] = 1 - bg_weights[:, :]

    output = np.full(img.shape, 255, dtype=np.uint8)
    for ch in range(channels):
        output[:, :, ch] = (img_weights[:, :] * img[:, :, ch]) + (
            bg_weights[:, :] * background[:, :, ch]
        )

    return output


for i in range(9):
    print(f"Processando imagem {i}...")

    img = cv2.imread(f"img/{i}.bmp").astype(np.uint8)
    background = cv2.imread("img/fundo.bmp").astype(np.uint8)
    background = cv2.resize(
        background, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA
    )

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cluster = HsvCluster()

    print("Clusterizando...")
    height, width, channels = hsv.shape
    for y in range(height):
        for x in range(width):
            cluster.calculate_pixel(hsv[y, x])

    cluster.calculate_colors_mean()

    if 10 > 60 - cluster.yellow.hue_mean:
        # Tem uns "amarelos esverdeados"
        sensitividade = (cluster.yellow.hue_mean - 30) / 2
    else:
        sensitividade = (cluster.green.hue_mean - 60) * 2

    lower_green = np.array([60 - sensitividade, 50, 50])
    upper_green = np.array([60 + sensitividade, 255, 255])

    print("Combinando...")
    output = simple_mask(img, background, hsv, lower_green, upper_green)

    print(f"Salvando imagem...")
    print("-----")
    cv2.imwrite(f"res/{i}.png", output)
