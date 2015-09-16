/*
 * TileCoder.cpp
 *
 *  Created on: Nov 18, 2011
 *      Author: igrondman@tudelft.net
 */

#include "acrl_tilecoding.h"

TileCoder::TileCoder(tileconfig * tcfg) : head(NULL) {
	cfg = tcfg;
	tiletotal = 1;

	int Ntiling0 = gsl_vector_get(cfg->Ntilings,0);
	int Ntiling1 = gsl_vector_get(cfg->Ntilings,1);
	int tiledim0 = gsl_vector_get(cfg->tiledim,0);
	int tiledim1 = gsl_vector_get(cfg->tiledim,1);

	double xshift = 0, yshift = 0;

	for (unsigned int i = 0; i < cfg->Ntilings->size; i++)
		tiletotal *= gsl_vector_get(cfg->Ntilings,i);

	for (int i = 0; i < Ntiling0; i++) {
		for (int j = 0; j < Ntiling1; j++) {
			// Calculate shifts
			if (cfg->shifts == 0) { // Uniform shifting
				xshift = -1.0/(tiledim0-1) + i*1.0/(Ntiling0*(tiledim0-1));
				yshift = -1.0/(tiledim1-1) + j*1.0/(Ntiling1*(tiledim1-1));
			}
			else if (cfg->shifts == 1) { // Random shifting
				// TODO implement ramdom shifting
			}
			// Add the actual tiling
			this->addTiling(tiledim0,tiledim1,xshift,yshift);
		}
	}
}

void TileCoder::getOutput(const gsl_vector *input, gsl_vector *output)
{
	// First convert input to the interval [0,1]
	gsl_vector * scaledin = scaleInput(input);

	Tiling * point = head;
	double outputsum = 0;
	while (point != NULL) {
		outputsum += point->getOutput(scaledin);
		point = point->nexttiling;
	}

	gsl_vector_set(output,0,outputsum/tiletotal);

	gsl_vector_free(scaledin);
}

void TileCoder::getOutput(const gsl_vector *input, gsl_vector *output, gsl_matrix *beta) {}

void TileCoder::reset()
{
	Tiling * point = head;
	while (point != NULL) {
		gsl_matrix_set_zero(point->tiling);
		gsl_matrix_set_zero(point->eligibility);
		point = point->nexttiling;
	}
}

void TileCoder::resetEligibility()
{
	Tiling * point = head;
	while (point != NULL) {
		gsl_matrix_set_zero(point->eligibility);
		point = point->nexttiling;
	}
}

void TileCoder::update(double error, gsl_vector * input, gsl_vector * output)
{
	gsl_vector * temperr = gsl_vector_alloc(1);
	gsl_vector_set(temperr,0,error);

	update(temperr,input,output);

	gsl_vector_free(temperr);
}

void TileCoder::update(gsl_vector * err, gsl_vector *input, gsl_vector *output)
{
	// TODO TileCoder must support multiple outputs!

	gsl_vector * scaledin = scaleInput(input);
	gsl_matrix * update = gsl_matrix_alloc(gsl_vector_get(cfg->tiledim,0),gsl_vector_get(cfg->tiledim,1));

	Tiling * point = head;
	while (point != NULL) {
		// Discount eligibility trace of tiling
		gsl_matrix_scale(point->eligibility,cfg->gamma*cfg->lambda);

		// Set eligibility for appropriate tile to 1
		point->setElig(scaledin);

		gsl_matrix_memcpy(update,point->eligibility);
		gsl_matrix_scale(update,gsl_vector_get(err,0)*cfg->alpha); // TODO This is still for single output
		gsl_matrix_add(point->tiling,update);
		gsl_vector_view boundrow = gsl_matrix_row(cfg->bound,0);
		sat(point->tiling, &boundrow.vector);

		// Point to next tiling
		point = point->nexttiling;
	}

	gsl_vector_free(scaledin);
	gsl_matrix_free(update);
}

TileCoder::~TileCoder()  {
	Tiling * point = head, * next;
	while (point != NULL) {
		next = point->nexttiling;
		delete point;
		point = next;
	}
}

gsl_vector *TileCoder::scaleInput(const gsl_vector *input)
{
	gsl_vector * scaledin = gsl_vector_alloc(input->size);

	double xrange = gsl_matrix_get(cfg->ssdims,0,1)-gsl_matrix_get(cfg->ssdims,0,0);
	double yrange = gsl_matrix_get(cfg->ssdims,1,1)-gsl_matrix_get(cfg->ssdims,1,0);

	gsl_vector_set(scaledin,0,satdouble((gsl_vector_get(input,0)-gsl_matrix_get(cfg->ssdims,0,0))/xrange,0.0,1.0));
	gsl_vector_set(scaledin,1,satdouble((gsl_vector_get(input,1)-gsl_matrix_get(cfg->ssdims,1,0))/yrange,0.0,1.0));

	return scaledin;
}

void TileCoder::getAction(const gsl_vector *input, const gsl_vector *output, gsl_vector *action)
{
}

void TileCoder::sat(gsl_matrix * satmatrix, gsl_vector * bounds)
{
	for (unsigned int k = 0; k < satmatrix->size1; k++) {
		for (unsigned int m = 0; m < satmatrix->size2; m++) {
			gsl_matrix_set(satmatrix,k,m,max(min(gsl_matrix_get(satmatrix,k,m),gsl_vector_get(bounds,1)),gsl_vector_get(bounds,0)));
		}
	}
}
