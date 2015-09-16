/*
 * solvers.h
 *
 *  Created on: Nov 16, 2011
 *      Author: igrondman@tudelft.net
 */

#ifndef ACRL_SOLVERS_H_
#define ACRL_SOLVERS_H_

#include <stdio.h>

#include <gsl/gsl_blas.h>

void rk4_ti(void (*odefun)(gsl_vector*, gsl_vector*), gsl_vector *, gsl_vector *, gsl_matrix *); // Runge-Kutta 4th order time-invariant

#endif /* ACRL_SOLVERS_H_ */
