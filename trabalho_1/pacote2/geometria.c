/*============================================================================*/
/* GEOMETRIA                                                                  */
/*----------------------------------------------------------------------------*/
/* Autor: Bogdan T. Nassu - nassu@dainf.ct.utfpr.edu.br                       */
/*============================================================================*/
/** Tipos e fun��es para representar entidades geom�tricas. */
/*============================================================================*/

#include "geometria.h"

/*============================================================================*/
/* COORDENADA                                                                 */
/*============================================================================*/
/** Apenas uma fun��o �til para "criar" uma inst�ncia do tipo Coordenada.
 *
 * Par�metros: int x: valor no eixo x.
 *             int y: valor no eixo y.
 *
 * Valor de retorno: a Coordenada criada. */

Coordenada criaCoordenada (int x, int y)
{
    Coordenada c;
    c.x = x;
    c.y = y;
    return (c);
}

/*============================================================================*/
/* RETANGULO                                                                  */
/*============================================================================*/
/** Apenas uma fun��o �til para "criar" uma inst�ncia do tipo Retangulo.
 *
 * Par�metros: int c: posi��o y do lado superior.
 *             int b: posi��o y do lado inferior.
 *             int e: posi��o x do lado esquerdo.
 *             int d: posi��o x do lado direito.
 *
 * Valor de retorno: o Retangulo criado. */

Retangulo criaRetangulo (int c, int b, int e, int d)
{
    Retangulo r;
    r.c = c;
    r.b = b;
    r.e = e;
    r.d = d;
    return (r);
}

/*============================================================================*/
