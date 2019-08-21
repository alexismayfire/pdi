/*============================================================================*/
/* Exemplo: segmentação de uma imagem em escala de cinza.                     */
/*----------------------------------------------------------------------------*/
/* Autor: Bogdan T. Nassu                                                     */
/* Universidade Tecnológica Federal do Paraná                                 */
/*============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pdi.h"

/*============================================================================*/

#define INPUT_IMAGE "arroz.bmp"

// TODO: ajuste estes parâmetros!
#define NEGATIVO 0
#define THRESHOLD 0.4f
#define ALTURA_MIN 1
#define LARGURA_MIN 1
#define N_PIXELS_MIN 1

/*============================================================================*/

typedef struct
{
    float label;
    Retangulo roi;
    int n_pixels;

} Componente;

/*============================================================================*/

void binariza (Imagem* in, Imagem* out, float threshold);
int rotula (Imagem* img, Componente** componentes, int largura_min, int altura_min, int n_pixels_min);

/*============================================================================*/

int main ()
{
    int i;

    // Abre a imagem em escala de cinza, e mantém uma cópia colorida dela para desenhar a saída.
    Imagem* img = abreImagem (INPUT_IMAGE, 1);
    if (!img)
    {
        printf ("Erro abrindo a imagem.\n");
        exit (1);
    }

    Imagem* img_out = criaImagem (img->largura, img->altura, 3);
    cinzaParaRGB (img, img_out);

    // Segmenta a imagem.
    if (NEGATIVO)
        inverte (img, img);
    binariza (img, img, THRESHOLD);
    salvaImagem (img, "01 - binarizada.bmp");

    Componente* componentes;
    int n_componentes;
    clock_t tempo_inicio = clock ();
    n_componentes = rotula (img, &componentes, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN);
    clock_t tempo_total = clock () - tempo_inicio;

    printf ("Tempo: %d\n", (int) tempo_total);
    printf ("%d componentes detectados.\n", n_componentes);

    // Mostra os objetos encontrados.
    for (i = 0; i < n_componentes; i++)
        desenhaRetangulo (componentes [i].roi, criaCor (1,0,0), img_out);
    salvaImagem (img_out, "02 - out.bmp");

    // Limpeza.
    free (componentes);
    destroiImagem (img_out);
    destroiImagem (img);
    return (0);
}

/*----------------------------------------------------------------------------*/
/** Binarização simples por limiarização.
 *
 * Parâmetros: Imagem* in: imagem de entrada. Se tiver mais que 1 canal,
 *               binariza cada canal independentemente.
 *             Imagem* out: imagem de saída. Deve ter o mesmo tamanho da
 *               imagem de entrada.
 *             float threshold: limiar.
 *
 * Valor de retorno: nenhum (usa a imagem de saída). */

void binariza (Imagem* in, Imagem* out, float threshold)
{
    if (in->largura != out->largura || in->altura != out->altura || in->n_canais != out->n_canais)
    {
        printf ("ERRO: binariza: as imagens precisam ter o mesmo tamanho e numero de canais.\n");
        exit (1);
    }

    // TODO: escreva o código desta função.
}

/*============================================================================*/
/* ROTULAGEM                                                                  */
/*============================================================================*/
/** Rotulagem usando flood fill. Marca os objetos da imagem com os valores
 * [0.1,0.2,etc].
 *
 * Parâmetros: Imagem* img: imagem de entrada E saída.
 *             Componente** componentes: um ponteiro para um vetor de saída.
 *               Supomos que o ponteiro inicialmente é inválido. Ele irá
 *               apontar para um vetor que será alocado dentro desta função.
 *               Lembre-se de desalocar o vetor criado!
 *             int largura_min: descarta componentes com largura menor que esta.
 *             int altura_min: descarta componentes com altura menor que esta.
 *             int n_pixels_min: descarta componentes com menos pixels que isso.
 *
 * Valor de retorno: o número de componentes conexos encontrados. */

int rotula (Imagem* img, Componente** componentes, int largura_min, int altura_min, int n_pixels_min)
{
    // TODO: escreva esta função.
	// Use a abordagem com flood fill recursivo.
	// Observe que o parâmetro 'componentes' é um ponteiro para um vetor, então a alocação dele deve ser algo como:
	// *componentes = malloc (sizeof (Componente) * n);
	// Dependendo de como você fizer a sua implementação, pode ser também interessante alocar primeiro um vetor maior do que o necessário, ajustando depois o tamanho usando a função realloc.
    return (0);
}

/*============================================================================*/
