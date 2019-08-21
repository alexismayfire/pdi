/*============================================================================*/
/* GEOMETRIA                                                                  */
/*----------------------------------------------------------------------------*/
/* Autor: Bogdan T. Nassu - nassu@dainf.ct.utfpr.edu.br                       */
/*============================================================================*/
/** Tipos e fun��es para representar entidades geom�tricas. */
/*============================================================================*/

#ifndef __GEOMETRIA_H
#define __GEOMETRIA_H

/*============================================================================*/
/* Coordenadas de um ponto (x,y). */

typedef struct
{
	int x;
	int y;

} Coordenada;

Coordenada criaCoordenada (int x, int y);

/*----------------------------------------------------------------------------*/
/* Coordenadas de um ret�ngulo. */

typedef struct
{
    int c; // Cima.
    int b; // Baixo.
    int e; // Esquerda.
    int d; // Direita.

} Retangulo;

Retangulo criaRetangulo (int c, int b, int e, int d);

/*============================================================================*/
#endif /* __GEOMETRIA_H */
