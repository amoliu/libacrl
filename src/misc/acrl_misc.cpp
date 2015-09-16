/*
 * rlmisc.cpp
 *
 *  Created on: Nov 18, 2011
 *      Author: igrondman@tudelft.net
 */

#include "acrl_misc.h"

void igsl_vector_append(gsl_vector * v1, gsl_vector * v2, gsl_vector * append)
{
	size_t v1size = v1->size;
	size_t v2size = v2->size;
	gsl_vector_view append1 = gsl_vector_subvector(append,0,v1size);
	gsl_vector_view append2 = gsl_vector_subvector(append,v1size,v2size);

	gsl_vector_memcpy(&append1.vector,v1);
	gsl_vector_memcpy(&append2.vector,v2);
}

void igsl_vector_addElement(gsl_vector * v1, double add, gsl_vector * append)
{
	size_t v1size = v1->size;
	gsl_vector_view append1 = gsl_vector_subvector(append,0,v1size);

	gsl_vector_memcpy(&append1.vector,v1);
	gsl_vector_set(append,v1size,add);
}

void sat(gsl_vector * input, gsl_matrix *bounds, gsl_vector * output)
{
	for (unsigned int k = 0; k < input->size; k++) {
		gsl_vector_set(output,k,max(min(gsl_vector_get(input,k),gsl_matrix_get(bounds,k,1)),gsl_matrix_get(bounds,k,0)));
	}
}

void sat(gsl_matrix * satmatrix, gsl_matrix *bounds)
{
	for (unsigned int k = 0; k < satmatrix->size1; k++) { // Iterate over all rows that have to be saturated
		for (unsigned int m = 0; m < satmatrix->size2; m++) { // Iterate over all columns that have to be saturated
			gsl_matrix_set(satmatrix,k,m,max(min(gsl_matrix_get(satmatrix,k,m),gsl_matrix_get(bounds,m,1)),gsl_matrix_get(bounds,m,0)));
		}
	}
}

double satdouble(double input, double minval, double maxval)
{
	return max(min(input,maxval),minval);
}

//int saveState(gsl_matrix * store, gsl_vector * state, gsl_vector * input, int index)
//{
//	if (store->size1 != state->size + input->size || store->size2 < index)
//		return 1;
//
//	gsl_vector matpartx = gsl_matrix_subcolumn(store,index,0,state->size).vector;
//	gsl_vector matpartu = gsl_matrix_subcolumn(store,index,state->size,input->size).vector;
//	gsl_vector_memcpy(&matpartx,state);
//	gsl_vector_memcpy(&matpartu,input);
//	return 0;
//}

bool withinBounds(gsl_vector *input, gsl_matrix *inputbounds)
{
	for (unsigned int k = 0; k < input->size; k++) {
		if (gsl_vector_get(input,k) < gsl_matrix_get(inputbounds,k,0) || gsl_vector_get(input,k) > gsl_matrix_get(inputbounds,k,1))
			return false;
	}
	return true;
}

void wrap(gsl_vector *input, gsl_vector *wrap) {
	for (unsigned int k = 0; k < input->size; k++) {
		if (gsl_vector_get(wrap,k)) {
			double val = gsl_vector_get(input,k);
			double sign = 1.0;
			if (val < 0)
				sign = -1.0;
			gsl_vector_set(input,k,fmod(val+sign*M_PI,2.*M_PI)-sign*M_PI);
		}
	}
}

void wrap(gsl_matrix *input) {
	for (unsigned int k = 0; k < input->size1; k++) {
		double val = gsl_matrix_get(input,k,0);
		double sign = 1.0;
		if (val < 0)
			sign = -1.0;

		gsl_matrix_set(input,k,0,fmod(val+sign*M_PI,2*M_PI)-sign*M_PI);
	}
}
