/*
 * RadialBasis.h
 *
 *  Created on: Dec 5, 2011
 *      Author: igrondman@tudelft.net
 */

#ifndef ACRL_GAUSSRBF_H_
#define ACRL_GAUSSRBF_H_

#include<cmath>
#include<string>
#include<gsl/gsl_multifit.h>


#include "acrl_fapprox.h"
#include "acrl_misc.h"

typedef struct {
	gsl_matrix * ssdims;
	gsl_vector * Nrbfs;
	gsl_matrix * bound;
	int outdim;
	double alpha;
	double gamma;
	double lambda;
} rbfconfig;


class RadialBasis : public FunApprox {
	rbfconfig * cfg;
	int rbftotal;
	gsl_matrix * param;
	gsl_vector * eligibility;
	gsl_matrix * cpos;
	gsl_matrix * invcov;
public:
	RadialBasis(rbfconfig *, double);

	// Generate the output for a certain input
	void getOutput(const gsl_vector *, gsl_vector *);

	// Generate the output and gradient for a certain input
	void getOutput(const gsl_vector *, gsl_vector *, gsl_matrix *);

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

	// Write the parameter to a file
	void writeParam(string);

	~RadialBasis();
private:
	void getPhi(const gsl_vector *, gsl_vector *, gsl_matrix *);
};

#endif /* ACRL_GAUSSRBF_H_ */
