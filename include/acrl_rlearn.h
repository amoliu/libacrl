/*
 * rlearn.h
 *
 *  Created on: Nov 18, 2011
 *      Author: igrondman@tudelft.net
 */

#ifndef ACRL_RLEARN_H_
#define ACRL_RLEARN_H_

#include <gsl/gsl_blas.h>

typedef struct {
	int Niterations;
	int Nepisodes;
	int erate;
	double odetime;
	double gamma;
	double esigma;
	double episodelength;
	double Ts;
	gsl_vector * x0;
	gsl_vector * statewrapping;
	gsl_matrix * inputbound;
} rlconfig;

#endif /* ACRL_RLEARN_H_ */
