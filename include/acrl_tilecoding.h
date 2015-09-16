/*
 * TileCoder.h
 *
 *  Created on: Nov 18, 2011
 *      Author: igrondman@tudelft.net
 */

#ifndef ACRL_TILECODING_H_
#define ACRL_TILECODING_H_


#include<stdio.h>
#include<string>

#include "acrl_fapprox.h"
#include "acrl_misc.h"

using namespace std;

typedef struct {
	gsl_matrix * ssdims;
	gsl_vector * tiledim;
	gsl_vector * Ntilings;
	int shifts;
	gsl_vector * wrapping;
	gsl_matrix * bound;
	double alpha;
	double gamma;
	double lambda;
} tileconfig;

class Tiling {
	gsl_matrix * tiling;
	gsl_matrix * eligibility;
	double x_shift, y_shift;
	Tiling * nexttiling;
public:
	// Constructor
	Tiling(int size_x, int size_y, double x_shift, double y_shift, Tiling* next = NULL) : x_shift(x_shift), y_shift(y_shift), nexttiling(next) {
		tiling = gsl_matrix_calloc(size_x,size_y);
		eligibility = gsl_matrix_calloc(size_x,size_y);
	}

	~Tiling() {
		gsl_matrix_free(tiling);
		gsl_matrix_free(eligibility);
		//printf("Freeing memory of a tiling\n");
	}
private:
	double getOutput(gsl_vector * input) {
		int xpos, ypos;
		findIndex(input,xpos,ypos);
		return gsl_matrix_get(tiling,xpos,ypos);
	}

	void setElig(gsl_vector *input) {
		int xpos, ypos;
		findIndex(input,xpos,ypos);
		gsl_matrix_set(eligibility,xpos,ypos,1);
	}

	void findIndex(gsl_vector *input, int& xpos, int& ypos) {
		xpos = 0;
		ypos = 0;
		while (gsl_vector_get(input,0) > x_shift+(xpos+1)*1.0/(tiling->size1-1)) {
			xpos++;
		}
		while (gsl_vector_get(input,1) > y_shift+(ypos+1)*1.0/(tiling->size2-1)) {
			ypos++;
		}
	}

	friend class TileCoder;
};

class TileCoder: public FunApprox {
	tileconfig * cfg;
	int tiletotal;
	Tiling * head;
public:
	TileCoder(tileconfig *);

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

	// Update the tile coding
	void update(double, gsl_vector *, gsl_vector *);

	// Update the function approximator (generalized to multi-output)
	void update(gsl_vector *, gsl_vector *, gsl_vector *);

	void writeParam(string) {};

	~TileCoder();
private:
	void addTiling(int size_x, int size_y, double x_shift, double y_shift) {
		head = new Tiling(size_x, size_y, x_shift, y_shift, head);
	}

	gsl_vector * scaleInput(const gsl_vector *);

	void sat(gsl_matrix*, gsl_vector *);
};

#endif /* ACRL_TILECODING_H_ */
