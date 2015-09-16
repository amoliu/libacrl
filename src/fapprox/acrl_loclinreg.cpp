/*
 * LocLinReg.cpp
 *
 *  Created on: Nov 28, 2011
 *      Author: igrondman@tudelft.net
 */

#include "acrl_loclinreg.h"

LocLinReg::LocLinReg(llrconfig * lcfg) : head(NULL) {
	cfg = lcfg;
	mem = gsl_matrix_alloc(cfg->numsamp,cfg->indim+cfg->outdim+2);

	// Add two caches (one for current, and one for previous state)
	for (int k = 0; k < 2; k++) {
		this->addCache(cfg->indim,cfg->outdim);
	}

	// Initialise the LLR function approximator
	this->reset();
}

// Generate the output for a certain input
void LocLinReg::getOutput(const gsl_vector * input, gsl_vector * output) {
	gsl_matrix * beta = gsl_matrix_alloc(cfg->indim,cfg->outdim);
	getOutput(input,output,beta);
	gsl_matrix_free(beta);
}

// Reset the whole function approximator object
void LocLinReg::reset() {
	gsl_matrix_set_all(mem,GSL_NAN);
	fillpos = 0;
	this->resetEligibility();
}

// Reset the eligibility traces
void LocLinReg::resetEligibility() {
	gsl_vector_view eligcolumn = gsl_matrix_column(mem,cfg->indim+cfg->outdim+1);
	gsl_vector_set_all(&eligcolumn.vector,0);
}

void LocLinReg::update(double error, gsl_vector * input, gsl_vector * output)
{
	gsl_vector * temperr = gsl_vector_alloc(1);
	gsl_vector_set(temperr,0,error);

	update(temperr,input,output);

	gsl_vector_free(temperr);
}

// Update the function approximator
void LocLinReg::update(gsl_vector * err, gsl_vector *input, gsl_vector *output)
{
	// Discount the eligibility trace
	gsl_vector_view eligcolumn = gsl_matrix_column(mem,cfg->indim+cfg->outdim+1);
	gsl_vector_scale(&eligcolumn.vector,cfg->gamma*cfg->lambda);

	// Get the relevant indices of neighbours and set their eligibility to 1
	gsl_matrix * beta = gsl_matrix_alloc(cfg->indim,cfg->outdim);
	gsl_vector * tempout = gsl_vector_alloc(cfg->outdim);
	gsl_vector * ind = NULL;

	getOutput(input,tempout,beta,ind);
	gsl_matrix_free(beta);
	gsl_vector_free(tempout);

	if (ind != NULL) {
		for (unsigned int k = 0; k < ind->size; k++) {
			if ((int)gsl_vector_get(ind,k) != lastfill)
				gsl_matrix_set(mem,(int)gsl_vector_get(ind,k),cfg->indim+cfg->outdim+1,1.0);
		}
	}

	// Insert the newly available sample
	int insindex;
	if (output != NULL) {
		insindex = insert(input,output);
		gsl_matrix_set(mem,insindex,cfg->indim+cfg->outdim+1,1.0);
	}

	// Calculate the update
	gsl_matrix * update = gsl_matrix_alloc(cfg->numsamp,cfg->outdim);
	for (int k = 0; k < cfg->outdim; k++) {
		gsl_vector_view updatecolumn = gsl_matrix_column(update,k);
		gsl_vector_view eligcolumn = gsl_matrix_column(mem,cfg->indim+cfg->outdim+1);
		gsl_vector_memcpy(&updatecolumn.vector, &eligcolumn.vector); // TODO: k indexering moet ook in mem gebruikt...
		gsl_vector_scale(&updatecolumn.vector,gsl_vector_get(err,k));
	}
	gsl_matrix_scale(update,cfg->alpha);

	// Update the outputs of the memory with the calculated update
	gsl_matrix_view memoutputview = gsl_matrix_submatrix(mem,0,cfg->indim,mem->size1,cfg->outdim);
	gsl_matrix * outputmem = &memoutputview.matrix;
	gsl_matrix_add(outputmem,update);
	sat(outputmem,cfg->bound);

	gsl_matrix_free(update);
	if (ind != NULL)
		gsl_vector_free(ind);

	//printMem();
}

LocLinReg::~LocLinReg() {
	// Free the whole LLR memory
	gsl_matrix_free(mem);

	// Delete the cache
	LLRCache * point = head, * next;
	while (point != NULL) {
		next = point->nextcache;
		delete point;
		point = next;
	}
}

int LocLinReg::insert(const gsl_vector *input, gsl_vector *output)
{
	int index;
	gsl_vector * nmeasure = NULL;
	gsl_vector * ind = NULL;
	gsl_matrix * noutput = NULL;
	gsl_matrix * beta = gsl_matrix_alloc(cfg->indim,cfg->outdim);

	// Initialise a random number generator
	gsl_rng * rng = gsl_rng_alloc(gsl_rng_default);

	// Replace neighbours' outputs with model output
	gsl_vector * modelout = gsl_vector_alloc(output->size);
	getOutput(input, modelout, beta, ind, noutput, nmeasure);
	if (ind != NULL) {
		int toUpdate;
		for (unsigned int k = 0 ; k < ind->size; k++ ) {
			toUpdate = (int)gsl_vector_get(ind,k);
			// Replace neighbour's output with model output
			gsl_vector_view memrow = gsl_matrix_subrow(mem,toUpdate,cfg->indim,cfg->outdim);
			gsl_vector_view noutputrow = gsl_matrix_row(noutput,k);
			gsl_vector_memcpy(&memrow.vector,&noutputrow.vector);

			// Update redundancy measure of neighbour
			double newval = .95*gsl_matrix_get(mem,toUpdate,cfg->indim+cfg->outdim) + .05*gsl_vector_get(nmeasure,k);
			gsl_matrix_set(mem,toUpdate,cfg->indim+cfg->outdim,newval);
		}
	}

	// Calculate the redundancy measure for current sample
	double smeasure;
	if (cfg->measure == 1) { // Model misfit
		gsl_vector_sub(modelout,output); // modelout now contains error
		smeasure = pow(gsl_blas_dnrm2(modelout),2);
	}
	else if (cfg->measure == 2) { // Distance
		if (nmeasure != NULL) {
			double total = 0;
			for (unsigned int k = 0; k < nmeasure->size; k++) {
				total += gsl_vector_get(nmeasure,k);
			}
			smeasure = total/nmeasure->size;
		}
		else
			smeasure = gsl_blas_dnrm2(input);
	}
	else { // Random sample purging
		smeasure = 0;
	}

	// Calculate the insertion position for the new sample
	if (fillpos < cfg->numsamp) { // If the memory is not full yet, we simply add the sample
		index = fillpos;
		fillpos++;
	}
	else if (cfg->measure > 0) { // Replace most redundant sample
		gsl_vector_view memcolumn = gsl_matrix_column(mem,cfg->indim+cfg->outdim);
		index = gsl_vector_min_index(&memcolumn.vector);
	}
	else { // Replace random sample

		index = (int)(gsl_rng_uniform(rng)*cfg->numsamp);
	}

	// Remember index of this inserted sample
	lastfill = index;

	// Insert the sample and set the eligibility to zero
	gsl_vector_view inputview = gsl_matrix_subrow(mem,index,0,cfg->indim);
	gsl_vector_view outputview = gsl_matrix_subrow(mem,index,cfg->indim,cfg->outdim);
	gsl_vector_memcpy(&inputview.vector,input);
	gsl_vector_memcpy(&outputview.vector,output);
	gsl_matrix_set(mem,index,cfg->indim+cfg->outdim,smeasure);
	gsl_matrix_set(mem,index,cfg->indim+cfg->outdim+1,0);


	// Free used memory
	gsl_vector_free(ind);
	gsl_matrix_free(noutput);
	gsl_matrix_free(beta);
	gsl_vector_free(nmeasure);
	gsl_vector_free(modelout);
	gsl_rng_free(rng);

	return index;
}

// Generate the output and model for a certain input
void LocLinReg::getOutput(const gsl_vector * input, gsl_vector * output, gsl_matrix * beta) {
	gsl_vector * ind = NULL;

	getOutput(input, output, beta, ind);

	if (ind != NULL) {
		gsl_vector_free(ind);
	}
}

void LocLinReg::getOutput(const gsl_vector * input, gsl_vector * output, gsl_matrix * beta, gsl_vector *& ind) {
	gsl_vector * measure = NULL;
	gsl_matrix * noutput = NULL;

	getOutput(input, output, beta, ind, noutput, measure);

	if (ind!= NULL) {
		gsl_vector_free(measure);
		gsl_matrix_free(noutput);
	}
}

void LocLinReg::getAction(const gsl_vector *input, const gsl_vector *output, gsl_vector *odein)
{
	gsl_vector * out = gsl_vector_alloc(output->size);
	gsl_vector_view actionview = gsl_vector_subvector(odein,input->size,1);
	gsl_vector * action = &actionview.vector;
	gsl_vector * delta = gsl_vector_alloc(action->size);

	gsl_matrix * beta = gsl_matrix_alloc(odein->size+1,output->size);
	gsl_matrix * cov = gsl_matrix_alloc(action->size,action->size);
	gsl_matrix * regressor = gsl_matrix_alloc(beta->size2,action->size);

	gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc(input->size,action->size);

	double chisq;

	// Get the output and gradient of f(x_k,u_{k-1})
//	gsl_vector_memcpy(&gsl_vector_subvector(odein,0,input->size).vector,input); // Do this if output has uses Taylor development around current state
	getOutput(odein,out,beta); // Here, out is just ignored, but used as mock variable

	// Calculate Refx - (beta_x*x + beta_b)
	gsl_vector_memcpy(out,output);
	gsl_matrix_view beta_xview = gsl_matrix_submatrix(beta,0,0,input->size,output->size);
	gsl_vector_view beta_bview = gsl_matrix_row(beta,beta->size1-1);
	gsl_blas_dgemv(CblasTrans,1.0,&beta_xview.matrix,input,1.0,&beta_bview.vector);
	gsl_vector_sub(out,&beta_bview.vector);

//	// Calculate Ref(x) - f(x,u_old)
//	gsl_vector_sub(out,output);
//	gsl_vector_scale(out,-1);

	// Get the partial derivative df(x,u)/du
	gsl_matrix_view beta_uview = gsl_matrix_submatrix(beta,input->size,0,action->size,beta->size2);
	gsl_matrix_transpose_memcpy(regressor,&beta_uview.matrix);

	// Perform the regression
	gsl_multifit_linear(regressor, out, action, cov, &chisq, work);
//	gsl_multifit_linear(regressor, out, delta, cov, &chisq, work);
//
//	// Delta has to be added to the action
//	gsl_vector_add(action,delta);

	gsl_multifit_linear_free(work);
	gsl_matrix_free(cov);

	gsl_matrix_free(beta);
	gsl_vector_free(delta);
	gsl_matrix_free(regressor);
	gsl_vector_free(out);
}

void LocLinReg::setMem(gsl_matrix *preknowl)
{
}

// Generate the output for a certain input and add additional information like model, indices of neighbours etc.
void LocLinReg::getOutput(const gsl_vector * input, gsl_vector * output, gsl_matrix * beta, gsl_vector *& ind, gsl_matrix *& noutput, gsl_vector *& nmeasure) {
	// If we have this input in cache, we should not recalculate everything
	LLRCache * point = head;
	while (point != NULL) {
		if (point->getCache(input,output,beta,ind,noutput,nmeasure))
			return;
		else
			point = point->nextcache;
	}

	// If no samples are present, return immediately after setting output to zero
	if (fillpos == 0) {
		gsl_vector_set_all(output,0);
		gsl_matrix_set_all(beta,0);

		// No cache has to be set as there were no neighbours
		return;
	}

	// Pre-allocate memory for matrices/vectors used in this procedure
	gsl_matrix * diff = gsl_matrix_alloc(cfg->numsamp,cfg->indim);
	gsl_matrix * inputm = gsl_matrix_alloc(1,cfg->indim);
	gsl_matrix * ones = gsl_matrix_alloc(cfg->numsamp,1);
	gsl_matrix * diffcopy = gsl_matrix_alloc(cfg->numsamp,cfg->indim);
	gsl_matrix * localbeta = gsl_matrix_alloc(input->size+1,output->size); // localbeta also has a bias term!
	gsl_vector * minbounds = gsl_vector_alloc(cfg->indim);
	gsl_vector * maxbounds = gsl_vector_alloc(cfg->indim);
	gsl_vector * inputaug = gsl_vector_alloc(cfg->indim+1);
	double * dist = new double[fillpos];

	// Neighbour search is not done in a separate function as with the Matlab code due to complexity with return values
	// Calculate distances
	gsl_vector_view inputmview = gsl_matrix_row(inputm,0);
	gsl_vector_memcpy(&inputmview.vector,input);
	gsl_matrix_set_all(ones,1);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, ones, inputm, 0.0, diff);
	gsl_matrix_view meminputview = gsl_matrix_submatrix(mem,0,0,cfg->numsamp,cfg->indim);
	gsl_matrix_sub(diff,&meminputview.matrix);
	if (gsl_vector_get(cfg->wrapping,0)) {
			wrap(diff);
	}
	gsl_matrix_memcpy(diffcopy,diff);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,diffcopy,cfg->weighting,0.0,diff);


	for (int k = 0; k < fillpos; k++) {
		gsl_vector_view diffrow = gsl_matrix_row(diff,k);
		dist[k] = gsl_blas_dasum(&diffrow.vector);
	}

	// We can't have more neighbours than available samples
	int numneigh = min(fillpos,cfg->numneighbours);

	// Now that the number of neighbours is known, allocate more memory for the neighbour dependent variables
	gsl_matrix * X = gsl_matrix_alloc(numneigh,cfg->indim+1);
	gsl_matrix * Y = gsl_matrix_alloc(numneigh,cfg->outdim);
	size_t * neighbours = new size_t[numneigh];

	gsl_sort_smallest_index(neighbours,numneigh,dist,1,fillpos);
	// END OF NEIGHBOUR SEARCH

	// From this point onwards, the variable neighbours contains the indices to use

	for (int k = 0; k < numneigh; k++) {
		// Fill X
		gsl_matrix_view Xrow = gsl_matrix_submatrix(X,k,0,1,cfg->indim);
		gsl_matrix_view neighbourX = gsl_matrix_submatrix(mem,neighbours[k],0,1,cfg->indim);
		gsl_matrix_memcpy(&Xrow.matrix,&neighbourX.matrix);
		gsl_matrix_set(X,k,X->size2-1,1); // bias term

		// Fill Y
		gsl_matrix_view Yrow = gsl_matrix_submatrix(Y,k,0,1,cfg->outdim);
		gsl_matrix_view neighbourY = gsl_matrix_submatrix(mem,neighbours[k],cfg->indim,1,cfg->outdim);
		gsl_matrix_memcpy(&Yrow.matrix,&neighbourY.matrix);
	}

	// Wrapping code
	gsl_vector * altinput = gsl_vector_alloc(cfg->indim);
	gsl_vector_memcpy(altinput,input);
	if (gsl_vector_get(cfg->wrapping,0)) {
		for (unsigned int k = 0; k < X->size1 ; k++) {
			gsl_matrix_set(X,k,0,gsl_matrix_get(X,k,0)-gsl_vector_get(input,0));
			gsl_vector_view Xrow = gsl_matrix_subrow(X,k,0,cfg->indim);
			wrap(&Xrow.vector,cfg->wrapping);
		}

		gsl_vector_set(altinput,0,0);
	}

	// Calculate minbounds/maxbounds
	for (int k = 0; k < cfg->indim; k++) {
		gsl_vector_view Xcolumn = gsl_matrix_column(X,k);
		gsl_vector_set(minbounds,k,gsl_vector_min(&Xcolumn.vector));
		gsl_vector_set(maxbounds,k,gsl_vector_max(&Xcolumn.vector));
	}

	gsl_vector_sub(minbounds,altinput); // minbounds has positive elements if input is outside
	gsl_vector_sub(maxbounds,altinput); // maxbounds has negative elements if input is outside

	/* TEST CODE TO REMOVE "ROBUSTNESS IN TIME" */
	if (cfg->indim == 3) {
		gsl_vector_set(minbounds,2,-1);
		gsl_vector_set(maxbounds,2,1);
	}
	/* END OF TEST CODE */

	// Robustness code
	if (cfg->robust && (!gsl_vector_isneg(minbounds) || !gsl_vector_ispos(maxbounds))) {
		ind = gsl_vector_alloc(1);
		nmeasure = gsl_vector_alloc(1);
		noutput = gsl_matrix_alloc(1,cfg->outdim);
		gsl_vector_set(ind,0,neighbours[0]);
		gsl_vector_set(nmeasure,0,dist[neighbours[0]]);
		gsl_matrix_view Ytoprow = gsl_matrix_submatrix(Y,0,0,1,cfg->outdim);
		gsl_matrix_memcpy(noutput,&Ytoprow.matrix);

		gsl_matrix_set_all(localbeta,0);
		gsl_vector_view lbetarow = gsl_matrix_row(localbeta,cfg->indim);
		gsl_vector_view Yrow = gsl_matrix_row(Y,0);
		gsl_vector_memcpy(&lbetarow.vector,&Yrow.vector);
		gsl_vector_memcpy(output,&Yrow.vector);

		// Set the cache
		if (cfg->cache)
			setCache(input,output,localbeta,ind,noutput,nmeasure);

		// Leaving early means we have to free the memory here!
		gsl_matrix_free(diff);
		gsl_matrix_free(inputm);
		gsl_matrix_free(ones);
		gsl_matrix_free(diffcopy);
		gsl_matrix_free(localbeta);
		gsl_vector_free(minbounds);
		gsl_vector_free(maxbounds);
		gsl_vector_free(inputaug);
		delete [] dist;

		gsl_matrix_free(X);
		gsl_matrix_free(Y);
		delete [] neighbours;

		return;
	}
	else {
		ind = gsl_vector_alloc(numneigh);
		nmeasure = gsl_vector_alloc(numneigh);
		noutput = gsl_matrix_alloc(numneigh,cfg->outdim);

		// Solve the linear least squares problem
		gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc(numneigh,cfg->indim+1);
		gsl_matrix * cov = gsl_matrix_alloc(cfg->indim+1,cfg->indim+1);
		double chisq;

		for (int k = 0; k < cfg->outdim; k++) {
			if (X->size1 < X->size2) {
				gsl_matrix_set_all(localbeta,0);
			}
			else {
				gsl_vector_view Ycolumn = gsl_matrix_column(Y,k);
				gsl_vector_view betacolumn = gsl_matrix_column(localbeta,k);
				gsl_multifit_linear(X, &Ycolumn.vector, &betacolumn.vector, cov, &chisq, work);
			}
		}

		gsl_multifit_linear_free(work);
		gsl_matrix_free(cov);
	}

	// Calculate the output of the LLR model
	gsl_vector_view inputaugview = gsl_vector_subvector(inputaug,0,cfg->indim);
	gsl_vector_memcpy(&inputaugview.vector,altinput);
	gsl_vector_set(inputaug,cfg->indim,1);
	gsl_blas_dgemv(CblasTrans,1.0,localbeta,inputaug,0.0,output);

	// Calculate the output for the neighbours
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,X,localbeta,0.0,noutput);

	// Calculate the redundancy measure for the neighbours
	gsl_matrix_sub(Y,noutput); // Y now contains difference between original and model output
	for (int k = 0; k < numneigh; k ++) {
		gsl_vector_set(ind,k,neighbours[k]);
		switch (cfg->measure)
		{
			case 1:
			{
				gsl_vector_view Yrow = gsl_matrix_row(Y,k);
				gsl_vector_set(nmeasure,k,pow(gsl_blas_dnrm2(&Yrow.vector),2)); break;
			}
			case 2: gsl_vector_set(nmeasure,k,dist[neighbours[k]]); break;
			default: gsl_vector_set(nmeasure,k,0); break;
		}
	}
	// Copy localbeta to beta, dropping the bias part
	if (beta != NULL) {
		if (beta->size1 == localbeta->size1-1) {
			gsl_matrix_view localbetaview = gsl_matrix_submatrix(localbeta,0,0,beta->size1,beta->size2);
			gsl_matrix_memcpy(beta,&localbetaview.matrix);
		}
		if (beta->size1 == localbeta->size1) {
			gsl_matrix_memcpy(beta,localbeta);
		}
	}

	// Set cache
	if (cfg->cache)
		setCache(input,output,localbeta,ind,noutput,nmeasure);

	// Free memory used
	gsl_matrix_free(diff);
	gsl_matrix_free(inputm);
	gsl_matrix_free(ones);
	gsl_matrix_free(diffcopy);
	gsl_matrix_free(localbeta);
	gsl_vector_free(minbounds);
	gsl_vector_free(maxbounds);
	gsl_vector_free(inputaug);
	gsl_vector_free(altinput);
	delete [] dist;

	gsl_matrix_free(X);
	gsl_matrix_free(Y);
	delete [] neighbours;
}

void LocLinReg::writeParam(string filename)
{
	FILE * outputfile = fopen(filename.c_str(),"ab");
	gsl_matrix_view meminout = gsl_matrix_submatrix(mem,0,0,cfg->numsamp,cfg->indim+cfg->outdim);
	gsl_matrix_fwrite(outputfile,&meminout.matrix);
	fclose(outputfile);
}

void LocLinReg::printMem()
{
	for (int k = 0 ; k < cfg->indim+cfg->outdim+2; k++) {
		for (int j = max(0,fillpos-10); j < max(fillpos,10) ; j ++) {
			printf("%10.4f ",gsl_matrix_get(mem,j,k));
		}
		printf("\n");
	}
	printf("\n");
}
