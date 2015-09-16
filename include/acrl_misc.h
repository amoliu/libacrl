/*
 * rlmisc.h
 *
 *  Created on: Nov 18, 2011
 *      Author: igrondman@tudelft.net
 */

#ifndef ACRL_MISC_H_
#define ACRL_MISC_H_

#include<algorithm>
#include<cmath>
#include<gsl/gsl_blas.h>
using namespace std;

void igsl_vector_append(gsl_vector * v1, gsl_vector * v2, gsl_vector * append);

void igsl_vector_addElement(gsl_vector * v1, double add, gsl_vector * append);

void sat(gsl_vector*, gsl_matrix*, gsl_vector*);

void sat(gsl_matrix *, gsl_matrix *);

double satdouble(double, double, double);

bool withinBounds(gsl_vector*, gsl_matrix*);

void wrap(gsl_vector*, gsl_vector*);

void wrap(gsl_matrix*);

#endif /* ACRL_MISC_H_ */
