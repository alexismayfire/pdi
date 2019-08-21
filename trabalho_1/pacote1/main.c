/*============================================================================*/
/* Exemplos: manipulação básica de imagens.                                   */
/*----------------------------------------------------------------------------*/
/* Autor: Bogdan T. Nassu                                                     */
/* Universidade Tecnológica Federal do Paraná                                 */
/*============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pdi.h"

/*============================================================================*/

int main ()
{
    Imagem* img;
    Imagem* canais [3];
    int i, j, k;

    //-------------------------------------------------------------------------
    // Exemplo 0: abrindo e salvando.
    img = abreImagem ("eyehand.bmp", 1); // Imagem em escala de cinza.
    salvaImagem (img, "exemplo0-1.bmp");
    destroiImagem (img);

    img = abreImagem ("flowers.bmp", 3); // Imagem colorida.
    salvaImagem (img, "exemplo0-2.bmp");
    destroiImagem (img);

    //-------------------------------------------------------------------------
    // Exemplo 1: negativo de uma imagem escala de cinza.
    img = abreImagem ("eyehand.bmp", 1);
    for (i = 0; i < img->altura; i++)
        for (j = 0; j < img->largura; j++)
            img->dados [0][i][j] = 1.0f - img->dados [0][i][j];

    salvaImagem (img, "exemplo1.bmp");
    destroiImagem (img);

    //-------------------------------------------------------------------------
    // Exemplo 2: negativo de uma imagem colorida.
    img = abreImagem ("flowers.bmp", 3);
    for (i = 0; i < img->altura; i++)
        for (j = 0; j < img->largura; j++)
            for (k = 0; k < 3; k++)
                img->dados [k][i][j] = 1.0f - img->dados [k][i][j];

    salvaImagem (img, "exemplo2.bmp");
    destroiImagem (img);

    //-------------------------------------------------------------------------
    // Exemplo 3: separando os canais de uma imagem colorida.
    img = abreImagem ("flowers.bmp", 3);
    for (i = 0; i < 3; i++)
        canais [i] = criaImagem (img->largura, img->altura, 1);

    for (i = 0; i < img->altura; i++)
        for (j = 0; j < img->largura; j++)
            for (k = 0; k < 3; k++)
                canais [k]->dados [0][i][j] = img->dados [k][i][j];

    salvaImagem (canais [0], "exemplo3-r.bmp");
    salvaImagem (canais [1], "exemplo3-g.bmp");
    salvaImagem (canais [2], "exemplo3-b.bmp");

    destroiImagem (img);
    for (i = 0; i < 3; i++)
        destroiImagem (canais [i]);

    //-------------------------------------------------------------------------
    // Exemplo 4: manipulando uma imagem em escala de cinza.
    img = criaImagem (256, 256, 1);

    for (i = 0; i < img->altura/2; i++)
        for (j = 0; j < img->largura/2; j++)
        {
            float dist = sqrtf (i*i + j*j); // Distância do pixel (j,i) até a origem.
            float valor = dist / (img->altura/2.0f * sqrtf (2)); // Proporção de dist para a distância até o centro da imagem.

            // Replica 4 vezes, espelhando.
            img->dados [0][i][j] = valor;
            img->dados [0][img->altura-1-i][j] = valor;
            img->dados [0][i][img->largura-1-j] = valor;
            img->dados [0][img->altura-1-i][img->largura-1-j] = valor;
        }

    salvaImagem (img, "exemplo4.bmp");
    destroiImagem (img);

    //-------------------------------------------------------------------------
    // Exemplo 5: manipulando uma imagem colorida.
     img = abreImagem ("flowers.bmp", 3);

    for (i = 0; i < img->altura; i++)
        for (j = 0; j < img->largura; j++)
        {
            img->dados [0][i][j] = i / (float) img->largura * img->dados [0][i][j]; // Faz o canal R "surgir" conforme vamos para baixo.
            img->dados [1][i][j] = j / (float) img->largura * img->dados [1][i][j]; // Faz o canal G "surgir" conforme vamos para a esquerda.
        }

    salvaImagem (img, "exemplo5.bmp");
    destroiImagem (img);

    return (0);
}

/*============================================================================*/
