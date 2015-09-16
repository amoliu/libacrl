/*
 * acrl_simmodels.h
 *
 *  Created on: Nov 16, 2011
 *      Author: igrondman@tudelft.net
 */

#ifndef ACRL_SIMMODELS_H_
#define ACRL_SIMMODELS_H_

#include <gsl/gsl_blas.h>
#include <math.h>
#include <stdio.h>

// Physical constants
#define CONS_GRAVITY 9.81	// Gravity constant

// Inverted pendulum constants
#define PEND_INERT 	1.91e-4	// Inertia
#define PEND_MASS  	5.5e-2	// Mass
#define PEND_LENGTH 4.2e-2	// Pendulum length
#define PEND_VDAMP 	3e-6	// Viscous damping
#define PEND_TORQUE 5.36e-2	// Torque constant
#define PEND_ROTOR  9.5  	// Rotor resistance

void invpend(gsl_vector *in, gsl_vector *out);
void twolink(gsl_vector *in, gsl_vector *out);
void twolinksym(gsl_vector *in, gsl_vector *out);
void twolinknog(gsl_vector *in, gsl_vector *out);

#endif /* ACRL_SIMMODELS_H_ */
