/*
 * solvers.cpp
 *
 *  Created on: Nov 16, 2011
 *      Author: igrondman@tudelft.net
 */

#include "acrl_solvers.h"

void rk4_ti(void(*odefun)(gsl_vector*, gsl_vector*), gsl_vector *tspan, gsl_vector *y0, gsl_matrix *yout)
{
	// Declare four vectors that will hold intermediate delta values
	gsl_vector *k1, *k2, *k3, *k4, *yprev, *temp;
	k1 = gsl_vector_calloc(y0->size);
	k2 = gsl_vector_calloc(y0->size);
	k3 = gsl_vector_calloc(y0->size);
	k4 = gsl_vector_calloc(y0->size);
	temp = gsl_vector_alloc(y0->size);

	// First column contains initial value
	gsl_matrix_set_col(yout,0,y0);

	//gsl_matrix_fprintf(stdout,yout,"%f");

	// Fill rest of the matrix with state at each point in time
	for (int k = 1; k < (int)tspan->size; k++)
	{
		// Get a pointer to the previous y value
		gsl_vector_view yprevview = gsl_matrix_column(yout,k-1);
		yprev = &yprevview.vector;

		// Calculate the time step to take
		double h = tspan->data[k]-tspan->data[k-1];

		// Calculate the 4 deltas needed in the RK4-method
		odefun(yprev,k1);

		gsl_vector_memcpy(temp,yprev);
		gsl_blas_daxpy(.5*h,k1,temp);
		odefun(temp,k2);

		gsl_vector_memcpy(temp,yprev);
		gsl_blas_daxpy(.5*h,k2,temp);
		odefun(temp,k3);

		gsl_vector_memcpy(temp,yprev);
		gsl_blas_daxpy(h,k3,temp);
		odefun(temp,k4);

		// Calculate the grand total delta (using k1 to add everything)
		gsl_vector_add(k1,k4);
		gsl_blas_daxpy(2.,k2,k1);
		gsl_blas_daxpy(2.,k3,k1);
		gsl_vector_memcpy(temp,yprev);
		gsl_blas_daxpy(h/6.,k1,temp);

		gsl_matrix_set_col(yout,k,temp);

	}
	gsl_vector_free(k1);
	gsl_vector_free(k2);
	gsl_vector_free(k3);
	gsl_vector_free(k4);
	gsl_vector_free(temp);
}
