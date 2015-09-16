/*
 * acrl_simmodels.cpp
 *
 *  Created on: Nov 16, 2011
 *      Author: igrondman@tudelft.net
 */

#include "acrl_simmodels.h"

// Inverted pendulum
void invpend(gsl_vector *in, gsl_vector *out)
{
	gsl_vector_set(out,0,gsl_vector_get(in,1));
	gsl_vector_set(out,1,(PEND_MASS*CONS_GRAVITY*PEND_LENGTH*sin(gsl_vector_get(in,0)) - (PEND_VDAMP + pow(PEND_TORQUE,2)/PEND_ROTOR)*gsl_vector_get(in,1) + PEND_TORQUE/PEND_ROTOR*gsl_vector_get(in,2))/PEND_INERT);
	gsl_vector_set(out,2,0);
}
