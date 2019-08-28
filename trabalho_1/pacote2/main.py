from datetime import datetime
import time

import cv2
import numpy as np

# Aqui usa as mesmas constantes que o código em C
INPUT_IMAGE = "arroz.bmp"
NEGATIVO = 0
THRESHOLD = 0.77
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 1

VERMELHO = np.array([0, 0, 1])


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


def salvar_imagem(nome, img):
    # Antes de salvar, como abrimos em valores normalizados e float (valores entre [0, 1])
    # agora precisa multiplicar por 255, seguindo as orientações do professor

    # Syntactic sugar, como img.shape é um vetor com 3 posições, podemos acessar assim.
    # O equivalente seria:

    # altura = img.shape[0]
    # largura = img.shape[1]
    # canais = img.shape[2]
    canais = None

    try:
        altura, largura, canais = img.shape
    except ValueError:
        # Imagem não tem 3 canais!
        altura, largura = img.shape

    # Cria uma matriz de ZEROS do tamanho da imagem
    saida = np.zeros(img.shape)

    # Preenche a matriz de saída com a imagem "Desnormalizando" ao multiplicar por 255
    for y in range(0, altura):
        for x in range(0, largura):
            if canais:
                for canal in range(0, canais):
                    saida[y][x][canal] = img[y][x][canal] * 255
            else:
                saida[y][x] = img[y][x] * 255

    # Salva a Imagem na pasta
    cv2.imwrite(nome, saida)


def inverte(img):
    # Explicação desse syntactic sugar na função salvar_imagem() acima!
    #
    # Mas aqui, definimos "_" como boa prática, porque não precisamos desse valor.
    # Porém, precisa "desempacotar" do vetor img.shape! Se fizer apenas assim:
    #
    # altura, largura = img.shape
    #
    # Vai dar uma exceção:
    # ValueError: too many values to unpack (expected 2)
    # img.shape é um vetor com 3 posições
    # e todas precisam ser retornadas
    altura, largura, _ = img.shape

    # Criamos uma nova matriz preenchida com zeros e do mesmo tamanho que a imagem
    img_invertida = np.zeros(img.shape)

    for y in range(0, altura):
        for x in range(0, largura):
            # Imagens abertas no OpenCV tem os canais invertidos
            # Ao invés de RGB, retorna BGR!
            # Outro syntactic sugar, como sabemos que "img[y][x]" é um vetor com 3 posições
            # (os valores dos 3 canais), podemos definir 3 variáveis, cada uma recebendo
            # o valor na ordem do vetor.
            # É equivalente a:
            # blue = img[y][x][0]
            # green = img[y][x][1]
            # red = img[y][x][2]
            b, g, r = img[y][x]

            # Isso aqui funciona corretamente apenas se a imagem for aberta normalizada!
            # Seguindo o código em C do professor, basta diminuir 1 dos valores dos canais
            img_invertida[y][x][0] = np.float(1) - b
            img_invertida[y][x][1] = np.float(1) - g
            img_invertida[y][x][2] = np.float(1) - r

    return img_invertida


def desenha_linha(p1, p2, img, cor):
    # Ver a explicação disso na função inverte() acima!
    altura, largura, _ = img.shape

    # Essa função vai receber dois objetos Coordenada, p1 e p2.
    # Cada coordenada (ponto) vai ter X e Y!
    # Só que ela é chamada por desenha_retangulo() e, da forma que as chamadas são feitas,
    # p1.x == p2.x ou p1.y == p2.y para qualquer das quatro chamadas...
    # No código em C o professor incluiu um TODO apenas nessa funçao, desenha_linha()
    # Mas creio que precise alterar as chamadas em desenha_retangulo() também!
    # Incluindo mais opções, talvez...

    if p1.x == p2.x:
        # Linha vertical
        #
        # Aqui é pra definir um início que seja maior que 0, pra não dar overflow na matriz
        # Se por alguma razão Y das coordenadas for menor que 0, inicio = 0 obrigatoriamente
        inicio = max(0, min(p1.y, p2.y))
        # Mesma coisa para o fim, obrigado a gerar um valor dentro da matriz.
        # Se Y for maior do que (altura da imagem - 1), fim = altura - 1
        fim = min(altura - 1, max(p1.y, p2.y))

        for y in range(inicio, fim):
            img[y][p1.x] = cor

    elif p1.y == p2.y:
        # Linha horizontal
        #
        # Mesma ideia que acima, só é invertido o min/max porque agora trata da largura
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
    # Ver a explicação disso na função inverte() acima!
    altura, largura, _ = img.shape

    # Essa função não foi comentada pelo professor, mas tem a seguinte lógica:
    # Cada if vai verificar se as coordenadas do retângulo estão dentro
    # da matriz da imagem. Por isso todas precisam ser maior que zero,
    # r.d e r.e precisam ser menores que a largura (estão no eixo X)
    # r.c e r.b precisam ser menores que a altura (estão no eixo Y)

    # Esses if são independentes no código em C, ou seja, vai SEMPRE testar as
    # 4 condições em cada chamada dessa função!
    # Apenas como observação, em Python o if segue o padrão abaixo.
    #
    # else if -> elif
    # && -> and
    # || -> or
    #
    # if condicao:
    #   print(condicao)
    # elif outra_condicao:
    #   print(outra_condicao)
    # else:
    #   print("Nenhuma das duas!")
    #
    # Não precisa separar os termos com (), apenas se quiser criar um isolamento
    # Em C isso seria da mesma forma:
    #
    # if condicao and (outra_condicao or ainda_outra_condicao):
    #   print("'condicao' é verdadeiro, e 'outra_condicao' ou 'ainda_outra_condicao' também!)
    #
    # No caso acima, 'condicao' e o resultado da expressão em () precisa ser verdadeiro.
    # Ou seja, apenas uma das duas variáveis entre parentêses precisaria ser verdadeira,
    # por causa do 'or'

    # Os dois primeiros parâmetros são objetos simples que estão sendo passados (Coordenada).
    # Cada objeto vai ter as propriedades X e Y.

    # Esquerda.
    if r.e >= 0 and r.e < largura:
        # Aqui, a primeira Coordenada vai ser:
        #
        # Coordenada.x = r.e
        # Coordenada.y = r.c
        #
        # E a segunda:
        #
        # Coordenada.x = r.e
        # Coordenada.y = r.b
        #
        # Ou seja, vamos ter dois pontos no plano da imagem, onde:
        # X = r.e
        # Y = [r.c, r.b]
        # Isso deve gerar uma linha VERTICAL, porque apenas Y está variando!
        img = desenha_linha(Coordenada(r.e, r.c), Coordenada(r.e, r.b), img, cor)

    # Direita.
    if r.d >= 0 and r.d < largura:
        # Aqui também, apenas Y vai variar:
        # X = r.d
        # Y = [r.c, r.b]
        # Isso deve gerar uma linha VERTICAL
        img = desenha_linha(Coordenada(r.d, r.c), Coordenada(r.d, r.b), img, cor)

    # Cima.
    if r.c >= 0 and r.c < altura:
        # Aqui, X vai variar e Y é fixo:
        # X = [r.e, r.d]
        # Y = r.c
        # Então, uma linha HORIZONTAL, porque apenas X está variando!
        img = desenha_linha(Coordenada(r.e, r.c), Coordenada(r.d, r.c), img, cor)

    # Baixo.
    if r.b >= 0 and r.b < altura:
        # Mesma coisa que acima, só X vai variar:
        # X = [r.e, r.d]
        # Y = r.b
        # Então, linha HORIZONTAL também!
        img = desenha_linha(Coordenada(r.e, r.b), Coordenada(r.d, r.b), img, cor)

    return img


def binariza(img, threshold):
    altura, largura, canais = img.shape

    # Cria uma imagem com apenas um canal
    img_out = np.zeros((altura, largura))
    
    # Percorre a imagem toda e analisa, se for maior que o threshold fica branco, senão fica preto
    for y in range(0, altura):
        for x in range(0, largura):
            for canal in range(0, canais):
                if img[y][x][canal] >= threshold:
                    img_out[y][x] = np.float(1.0)
                else:
                    img_out[y][x] = np.float(0)

    return img_out


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
            if img[y][x] == np.float(1):
                img[y][x] = -1

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


if __name__ == "__main__":
    # .astype(np.float) é para abrir os valores como float e não int
    # Dessa forma, é possível normalizar em seguida
    img = cv2.imread(INPUT_IMAGE).astype(np.float)
    # Normalizando os valores dos pixels para ficar no intervalo [0, 1]
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # img.shape contem altura, largura e canais em uma tupla: (altura, largura, canais)
    # A função np.zeros cria uma matriz preenchida com zeros no formato passado
    # Ou seja, [altura][largura][canais]
    # for i in range(0, 3):
    # Aqui é um syntactic sugar do Python, ao invés de iterar na matriz
    # é possível copiar os valores usando array slicing
    # Ref: https://stackoverflow.com/questions/509211/understanding-slice-notation
    # Sabendo que a matriz "img" tem três dimensões, e acessamos um pixel assim:
    #
    # pixel = img[y][x][canal]
    #
    # A construção abaixo é equivalente a:
    #
    # for y in range(0, altura):
    #   for x in range(0, largura):
    #       for canal in range(0, canais):
    #           img_out[y][x][canal] = img[y][x][0]
    #
    # Ou seja, estamos copiando apenas o valor do canal 0 nos 3 canais de img_out!
    # O professor executa isso na função cinzaParaRgb no código em C:
    #
    # int i, j, k;
    # for (i = 0; i < 3; i++)
    #   for (j = 0; j < in->altura; j++)
    #        for (k = 0; k < in->largura; k++)
    #            out->dados [i][j][k] = in->dados [0][j][k];
    # img_out[:, :, i] = img[:, :, 0]

    if NEGATIVO:
        img = inverte(img)

    img_out = binariza(img, THRESHOLD)
    salvar_imagem("01 - binarizada.bmp", img_out)

    tempo_inicio = datetime.now()
    # Aqui a função rotula() deve retornar dois elementos, em Python pode!
    componentes = rotula(img_out, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    tempo_total = datetime.now() - tempo_inicio

    # Esse 'f' na frente é pra interpolação de variáveis em strings, usando elas dentro de {}.
    # Disponível a partir do Python 3.6
    print(f"Tempo: {tempo_total}")
    print(f"Componentes detectados: {len(componentes)}")

    # Mostra os objetos encontrados
    for i in range(0, len(componentes)):
        img = desenha_retangulo(componentes[i].retangulo, img)

    salvar_imagem("02 - out.bmp", img)