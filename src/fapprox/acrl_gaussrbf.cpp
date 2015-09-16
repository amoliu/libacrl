/*
 * RadialBasis.cpp
 *
 *  Created on: Dec 5, 2011
 *      Author: igrondman@tudelft.net
 */

#include "acrl_gaussrbf.h"

RadialBasis::RadialBasis(rbfconfig * rcfg, double isect) : cfg(rcfg) {
	// Introduce a shorthand for the state size
	int statesize = cfg->ssdims->size1;

	// Calculate the total number of basis functions needed
	rbftotal = 1;
	int * rbfmod = new int[statesize];
	for (int k = 0; k < statesize; k++) {
		rbfmod[k] = rbftotal;
		rbftotal *= gsl_vector_get(cfg->Nrbfs,k);
	}

	// Allocate memory for parameters, eligibility traces and center positions
	param = gsl_matrix_alloc(rbftotal,cfg->outdim);
	eligibility = gsl_vector_alloc(rbftotal);
	cpos = gsl_matrix_alloc(rbftotal,statesize);
	invcov = gsl_matrix_alloc(statesize,statesize);

	// Calculate invcov
	for (int j = 0; j < statesize; j++) {
		double diff = (gsl_matrix_get(cfg->ssdims,j,1)-gsl_matrix_get(cfg->ssdims,j,0))/(gsl_vector_get(cfg->Nrbfs,j)-1);
		gsl_matrix_set(invcov,j,j,-8.0*log(isect)/pow(diff,2));
	}

	for (int i = 0; i < rbftotal; i++) {
		for (int j = 0; j < statesize; j++) {
			gsl_matrix_set(cpos,i,j,gsl_matrix_get(cfg->ssdims,j,0) + ((i/rbfmod[j]) % (int)gsl_vector_get(cfg->Nrbfs,j))*(gsl_matrix_get(cfg->ssdims,j,1)-gsl_matrix_get(cfg->ssdims,j,0))/(gsl_vector_get(cfg->Nrbfs,j)-1));
		}
	}

	delete [] rbfmod;

	// Reset the whole function approximator
	reset();
}

void RadialBasis::getOutput(const gsl_vector *input, gsl_vector *output)
{
	// Allocate memory
	gsl_vector * phi = gsl_vector_alloc(rbftotal);

	// Get the values of phi
	getPhi(input,phi,NULL);

	// Put inner product param*phi in output
	for (unsigned int k = 0; k < output->size; k++) {
		gsl_vector_view paramcolumn = gsl_matrix_column(param,k);
		gsl_blas_ddot(&paramcolumn.vector,phi,gsl_vector_ptr(output,k));
	}

	// Free allocated memory
	gsl_vector_free(phi);
}

void RadialBasis::getOutput(const gsl_vector *input, gsl_vector *output, gsl_matrix *beta)
{
	// Allocate memory
	gsl_vector * phi = gsl_vector_alloc(rbftotal);
	gsl_matrix * dphi = gsl_matrix_alloc(rbftotal,input->size);

	// Get the values of phi and derivatives
	getPhi(input,phi,dphi);

	// Calculate output
	for (unsigned int k = 0; k < output->size; k++) {
		gsl_vector_view paramcolumn = gsl_matrix_column(param,k);
		gsl_vector_view betacolumn = gsl_matrix_column(beta,k);

		// Put inner product param*phi in output
		gsl_blas_ddot(&paramcolumn.vector,phi,gsl_vector_ptr(output,k));

		// Put gradient in a column of the beta matrix
		gsl_blas_dgemv(CblasTrans,1.0,dphi,&paramcolumn.vector,0.0,&betacolumn.vector);
	}

	// Free allocated memory
	gsl_vector_free(phi);
	gsl_matrix_free(dphi);
}

void RadialBasis::reset()
{
	gsl_matrix_set_all(param,0);
	resetEligibility();
}

void RadialBasis::resetEligibility()
{
	gsl_vector_set_all(eligibility,0);
}

void RadialBasis::update(double err, gsl_vector *input, gsl_vector *output)
{
	gsl_vector * temperr = gsl_vector_alloc(1);
	gsl_vector_set(temperr,0,err);

	update(temperr,input,output);

	gsl_vector_free(temperr);
}

void RadialBasis::update(gsl_vector * err, gsl_vector *input, gsl_vector *output)
{
	// Allocate memory
	gsl_vector * phi = gsl_vector_alloc(rbftotal);
	gsl_vector * temp = gsl_vector_alloc(eligibility->size);

	// Get the current value of the phi's
	getPhi(input, phi, NULL);

	// Update eligibility trace
	gsl_vector_scale(eligibility,cfg->gamma*cfg->lambda); // Discount
	gsl_vector_add(eligibility,phi); // Add feature values

	// Update parameter vector
	for (unsigned int k = 0; k < output->size; k++) {
		gsl_vector_view paramcolumn = gsl_matrix_column(param,k);

		gsl_vector_memcpy(temp,eligibility);
		gsl_vector_scale(temp,cfg->alpha*gsl_vector_get(err,k));
		gsl_vector_add(&paramcolumn.vector,temp);
	}

	// Free allocated memory
	gsl_vector_free(temp);
	gsl_vector_free(phi);
}

void RadialBasis::writeParam(string filename) {
	FILE * outputfile = fopen(filename.c_str(),"ab");
	gsl_matrix_fwrite(outputfile,param);
	fclose(outputfile);

	outputfile = fopen("invcov.dat","wb");
	gsl_matrix_fwrite(outputfile,invcov);
	fclose(outputfile);

	outputfile = fopen("cpos.dat","wb");
	gsl_matrix_fwrite(outputfile,cpos);
	fclose(outputfile);
}

RadialBasis::~RadialBasis() {
	gsl_vector_free(eligibility);
	gsl_matrix_free(param);
	gsl_matrix_free(cpos);
}

void RadialBasis::getAction(const gsl_vector *input, const gsl_vector *output, gsl_vector *odein)
{
	// Allocate memory
	gsl_matrix * beta = gsl_matrix_alloc(odein->size,output->size);
	gsl_vector * out = gsl_vector_alloc(output->size);
	gsl_vector_view odeinput = gsl_vector_subvector(odein,odein->size-1,1);

	gsl_vector * action = &odeinput.vector;
	gsl_vector * delta = gsl_vector_alloc(action->size);
	gsl_matrix * regressor = gsl_matrix_alloc(beta->size2,action->size);
	gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc(input->size,action->size);
	gsl_matrix * cov = gsl_matrix_alloc(action->size,action->size);

	double chisq;

	// Get the output and gradient of f(x_k,u_{k-1})
	gsl_vector_view odestate = gsl_vector_subvector(odein,0,input->size);
	gsl_vector_memcpy(&odestate.vector,input); // Do this if output has uses Taylor development around current state
	getOutput(odein,out,beta);

	// Calculate Ref(x) - f(x,u_old)
	gsl_vector_sub(out,output);
	gsl_vector_scale(out,-1);

	// Take the input part of beta as the regressor and calculate delta u
	gsl_matrix_view betainput = gsl_matrix_submatrix(beta,input->size,0,action->size,beta->size2);
	gsl_matrix_transpose_memcpy(regressor,&betainput.matrix);
	gsl_multifit_linear(regressor, out, delta, cov, &chisq, work);

	// Delta u has to be added to the action
	gsl_vector_add(action,delta);

	// Free allocated memory
	gsl_matrix_free(beta);
	gsl_vector_free(out);
	gsl_vector_free(delta);
	gsl_matrix_free(regressor);
	gsl_multifit_linear_free(work);
	gsl_matrix_free(cov);
}

void RadialBasis::getPhi(const gsl_vector * input, gsl_vector *phi, gsl_matrix *dphi) {
	double total = 0;
	double val;
	gsl_vector * diff = gsl_vector_alloc(input->size);
	gsl_vector * result = gsl_vector_alloc(input->size);
	gsl_vector * dtotal = gsl_vector_calloc(input->size);

	for (unsigned int k = 0; k < phi->size; k++) {
		// Calculate the difference between input and center
		gsl_vector_memcpy(diff,input);
		gsl_vector_view cposrow = gsl_matrix_row(cpos,k);
		gsl_vector_sub(diff,&cposrow.vector);

		// Calculate diff' * invcov * diff and put the result in val
		gsl_blas_dsymv(CblasUpper,1.,invcov,diff,0.,result);
		gsl_blas_ddot(diff,result,&val);

		val = exp(-0.5*val);
		gsl_vector_set(phi,k,val);
		total+= val;

		if (dphi != NULL) { // If we want to calculate the derivative of phi with respect to x...
			// dphi(k,:) = invcov * (x-c)
			gsl_vector_view dphirow = gsl_matrix_row(dphi,k);
			gsl_vector_memcpy(&dphirow.vector,result);
			gsl_blas_daxpy(val,&dphirow.vector,dtotal);
		}
	}

	// Normalise phi
	gsl_vector_scale(phi,1.0/total);

	if (dphi != NULL) {
		// dphi = -phi*(dphi - dtotal/total)
		gsl_vector_scale(dtotal,1.0/total);

		for (unsigned int k = 0; k < input->size; k++) {
			gsl_vector_view dphicolumn = gsl_matrix_column(dphi,k);
			gsl_vector_add_constant(&dphicolumn.vector,-gsl_vector_get(dtotal,k));
			gsl_vector_mul(&dphicolumn.vector,phi);
		}
		gsl_matrix_scale(dphi,-1);
	}

	// Free allocated memory
	gsl_vector_free(diff);
	gsl_vector_free(dtotal);
	gsl_vector_free(result);
}
