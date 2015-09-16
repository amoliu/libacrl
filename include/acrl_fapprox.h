/*
 * FunApprox.h
 *
 *  Created on: Nov 18, 2011
 *      Author: igrondman@tudelft.net
 */

#ifndef FUNAPPROX_H_
#define FUNAPPROX_H_

#include<algorithm>
#include<gsl/gsl_blas.h>

using namespace std;

class FunApprox {
public:
	// Generate the output for a certain input
	virtual void getOutput(const gsl_vector *, gsl_vector *) = 0;

	// Generate the output and gradient for a certain input
	virtual void getOutput(const gsl_vector *, gsl_vector *, gsl_matrix *) = 0;

	// Generate an input based on a reference state
	virtual void getAction(const gsl_vector *, const gsl_vector *, gsl_vector *) = 0;

	// Reset the whole function approximator object
	virtual void reset() = 0;

	// Reset the eligibility traces
	virtual void resetEligibility() = 0;

	// Update the function approximator
	virtual void update(double, gsl_vector *, gsl_vector *) = 0;

	// Update the function approximator (generalized to multiple outputs)
	virtual void update(gsl_vector *, gsl_vector *, gsl_vector *) = 0;

	virtual void writeParam(string) = 0;

	virtual ~FunApprox() {};
};

#endif /* FUNAPPROX_H_ */
