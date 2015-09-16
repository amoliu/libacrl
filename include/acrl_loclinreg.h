/*
 * LocLinReg.h
 *
 *  Created on: Nov 28, 2011
 *      Author: igrondman@tudelft.net
 */

#ifndef ACRL_LOCLINREG_H_
#define ACRL_LOCLINREG_H_

#include<cmath>
#include<string>
#include<gsl/gsl_multifit.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_sort_double.h>
#include "acrl_fapprox.h"
#include "acrl_misc.h"

using namespace std;

typedef struct {
	int numsamp;
	int numneighbours;
	int indim, outdim;
	int robust, measure;
	bool cache;
	gsl_matrix * weighting;
	gsl_vector * wrapping;
	gsl_matrix * bound;
	double alpha;
	double gamma;
	double lambda;
} llrconfig;

class LLRCache {
	gsl_vector * input;
	gsl_vector * output;
	gsl_matrix * beta;
	gsl_vector * ind;
	gsl_matrix * noutput;
	gsl_vector * nmeasure;
	LLRCache * nextcache;
public:
	LLRCache(int indim, int outdim, LLRCache * next = NULL) : nextcache(next) {
		input = gsl_vector_alloc(indim);
		gsl_vector_set(input,0,100); // Prevent (0,0) from being matched to cache immediately
		output = gsl_vector_alloc(outdim);
		beta = gsl_matrix_alloc(indim+1,outdim); // Cache stores beta including bias term
	}

	~LLRCache() {
		gsl_vector_free(input);
		gsl_vector_free(output);
		gsl_matrix_free(beta);
		resetCache();
	}

private:
    void setCache(const gsl_vector * input, const gsl_vector * output, const gsl_matrix * beta, const gsl_vector * ind, const gsl_matrix * noutput, const gsl_vector * nmeasure) {
    	resetCache();
    	this->ind = gsl_vector_alloc(ind->size);
    	this->noutput = gsl_matrix_alloc(noutput->size1,noutput->size2);
    	this->nmeasure = gsl_vector_alloc(nmeasure->size);
    	gsl_vector_memcpy(this->input,input);
    	gsl_vector_memcpy(this->output,output);
    	gsl_matrix_memcpy(this->beta,beta);
    	gsl_vector_memcpy(this->ind,ind);
    	gsl_matrix_memcpy(this->noutput,noutput);
    	gsl_vector_memcpy(this->nmeasure,nmeasure);
    }

    int getCache(const gsl_vector * input, gsl_vector * output, gsl_matrix * beta, gsl_vector *& ind, gsl_matrix *& noutput, gsl_vector *& nmeasure) {
    	if (gsl_vector_equal(input,this->input)) {
    		ind = gsl_vector_alloc(this->ind->size);
    		noutput = gsl_matrix_alloc(this->noutput->size1,this->noutput->size2);
    		nmeasure = gsl_vector_alloc(this->nmeasure->size);
    		gsl_vector_memcpy(output,this->output);
    		if (beta->size1 == this->beta->size1) {
    			gsl_matrix_memcpy(beta,this->beta);
    		}
    		else {
    			gsl_matrix_view betaview = gsl_matrix_submatrix(this->beta,0,0,input->size,output->size);
    			gsl_matrix_memcpy(beta,&betaview.matrix);
    		}
    		gsl_vector_memcpy(ind,this->ind);
    		gsl_matrix_memcpy(noutput,this->noutput);
    		gsl_vector_memcpy(nmeasure,this->nmeasure);
    		return 1;
    	}
    	else return 0;
    }

    void resetCache() {
    	if (ind != NULL)
    		gsl_vector_free(ind);
    	if (noutput != NULL)
    		gsl_matrix_free(noutput);
    	if (nmeasure != NULL)
    		gsl_vector_free(nmeasure);
    	ind = NULL;
    	noutput = NULL;
    	nmeasure = NULL;
    }

    friend class LocLinReg;
};

class LocLinReg: public FunApprox {
	llrconfig * cfg;
	gsl_matrix * mem;
	LLRCache * head;
	int fillpos;
	int lastfill;
public:
	LocLinReg(llrconfig *);

	// Generate the output for a certain input
	void getOutput(const gsl_vector *, gsl_vector *);

	// Generate the output and gradient for a certain input
	void getOutput(const gsl_vector * input, gsl_vector * output, gsl_matrix * beta);

	// Generate an input based on a reference state
	void getAction(const gsl_vector *, const gsl_vector *, gsl_vector *);

	// Reset the whole function approximator object
	void reset();

	// Reset the eligibility traces
	void resetEligibility();

	// Update the function approximator
	void update(double, gsl_vector *, gsl_vector *);

	// Update the function approximator (generalized to multi-output)
	void update(gsl_vector *, gsl_vector *, gsl_vector *);

	void writeParam(string);

	// Temporary functions for design
	void setMem(gsl_matrix*);
	void printMem();

	~LocLinReg();
private:
	void addCache(int indim, int outdim) {
		head = new LLRCache(indim, outdim, head);
	}

	void setCache(const gsl_vector * input, const gsl_vector * output, const gsl_matrix * beta, const gsl_vector * ind, const gsl_matrix * noutput, const gsl_vector * nmeasure) {
		// Cycle caches, such that oldest is now at the tail and will be replaced
		LLRCache * toEnd = head;
		head = head->nextcache; // Set the head to the next element
		LLRCache * point = head;
		while (point->nextcache != NULL) // Scroll towards the last element
			point = point->nextcache;
		// Put the thing that was at the head at the end
		point->nextcache = toEnd;
		toEnd->nextcache = NULL;
		toEnd->setCache(input,output,beta,ind,noutput,nmeasure);
	}

	int insert(const gsl_vector *, gsl_vector *);



	void getOutput(const gsl_vector * input, gsl_vector * output, gsl_matrix * beta, gsl_vector *& ind);

	void getOutput(const gsl_vector * input, gsl_vector * output, gsl_matrix * beta, gsl_vector *& ind, gsl_matrix *& noutput, gsl_vector *& nmeasure);
};

#endif /* ACRL_LOCLINREG_H_ */
