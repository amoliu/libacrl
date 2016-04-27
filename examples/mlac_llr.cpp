/*
 * examplerun.cpp
 *
 *  Created on: Nov 18, 2011
 *      Author: igrondman@tudelft.net
 */

#include<fstream>
#include<time.h>

#include "acrl_actorcritic.h"
#include "acrl_loclinreg.h"
#include "acrl_rlearn.h"
#include "acrl_simmodels.h"

using namespace std;

double reward(const gsl_vector* x, const gsl_vector * u, const gsl_vector* xn);

// Allocating memory for Qrew in global scope
gsl_matrix * Qrew = gsl_matrix_calloc(2,2);

int main(int argc, char** argv) {
	// Allocate memories
	gsl_matrix * criticbound = gsl_matrix_alloc(1,2);
	gsl_matrix * processbound = gsl_matrix_alloc(2,2);
	gsl_matrix * proweighting = gsl_matrix_calloc(3,3);
	gsl_matrix * ubound = gsl_matrix_alloc(1,2);
	gsl_matrix * weighting = gsl_matrix_calloc(2,2);

	gsl_vector * nowrapping = gsl_vector_alloc(3);
	gsl_vector * wrapping = gsl_vector_alloc(2);
	gsl_vector * x0 = gsl_vector_alloc(2);

	double actor_alpha,critic_alpha, process_alpha, x1,x2;
	int actor_neigh, critic_neigh, process_neigh, actor_size, critic_size, process_size;
	char outputname[60];

	// General: set default learning rates
	actor_alpha = 0.05;
	critic_alpha = .3;
	process_alpha = 0;

	// LLR: set default neighbourhood sizes
	actor_neigh = 25;
	critic_neigh = 15;
	process_neigh = 10;

	// LLR: set default memory sizes
	actor_size = 2000;
	critic_size = 2000;
	process_size = 100;

	// Set default sizes of state space
	x1 = M_PI;
	x2 = 8*M_PI;

//	if (argc > 8) { // Parse the command line arguments
//		ref_alpha = atof(argv[1]);
//		critic_alpha = atof(argv[2]);
//		process_alpha = atof(argv[3]);
//		a_isect = atof(argv[4]);
//		c_isect = atof(argv[5]);
//		p_isect = atof(argv[6]);
//		r1 = atoi(argv[7]);
//		r2 = atoi(argv[7]);
//		r3 = atoi(argv[7]);
//		r4 = atoi(argv[7]);
//	}

//	if (argc > 4) { // Parse the command line arguments
//		actor_alpha = atof(argv[1]);
//		critic_alpha = atof(argv[2]);
//		actor_neigh = atoi(argv[3]);
//		critic_neigh = atoi(argv[4]);
//	}

	// Set the initial state
	gsl_vector_set(x0,0,M_PI);
	gsl_vector_set(x0,1,0);

	// Set the wrapping bit for each dimension
	gsl_vector_set(wrapping,0,1);
	gsl_vector_set(wrapping,1,0);

	// Saturated input
	gsl_matrix_set(ubound,0,0,-3);
	gsl_matrix_set(ubound,0,1,3);

	rlconfig * rlcfg = new rlconfig;
	rlcfg->Niterations = 1;
	rlcfg->Nepisodes = 600;
	rlcfg->gamma = 0.97;
	rlcfg->esigma = 1;
	rlcfg->erate = 3;
	rlcfg->episodelength = 3.;
	rlcfg->Ts = .03;
	rlcfg->odetime = rlcfg->Ts/10;
	rlcfg->x0 = x0;
	rlcfg->statewrapping = wrapping;
	rlcfg->inputbound = ubound;

	// Configure reward function matrix Q
	gsl_matrix_set(Qrew,0,0,5.);
	gsl_matrix_set(Qrew,1,1,.1);

	// Set the wrapping bit to zero for each dimension
	gsl_vector_set_all(nowrapping,0);

	// Set bounds on the output of the critic
	gsl_matrix_set(criticbound,0,0,-INFINITY);
	gsl_matrix_set(criticbound,0,1,INFINITY);

	// Set the bounds on the output of the processmodel
	gsl_matrix_set(processbound,0,0,-INFINITY);
	gsl_matrix_set(processbound,0,1,INFINITY);
	gsl_matrix_set(processbound,1,0,-INFINITY);
	gsl_matrix_set(processbound,1,1,INFINITY);

	llrconfig * actorcfg_llr = new llrconfig;
	llrconfig * criticcfg_llr = new llrconfig;
	llrconfig * processcfg_llr = new llrconfig;

	// Set the weighting matrix for actor and critic
	gsl_matrix_set(weighting,0,0,1);
	gsl_matrix_set(weighting,1,1,.1);

	// Set the weighting matrix for the process model
	gsl_matrix_set(proweighting,0,0,1);
	gsl_matrix_set(proweighting,1,1,.1);
	gsl_matrix_set(proweighting,2,2,1);

	// Actor configuration
	actorcfg_llr->numsamp = actor_size;
	actorcfg_llr->numneighbours = actor_neigh;
	actorcfg_llr->indim = 2;
	actorcfg_llr->outdim = 1;
	actorcfg_llr->robust = 1;
	actorcfg_llr->measure = 2;
	actorcfg_llr->wrapping = wrapping;
	actorcfg_llr->weighting = weighting;
	actorcfg_llr->gamma = rlcfg->gamma;
	actorcfg_llr->alpha = actor_alpha;
	actorcfg_llr->lambda = 0;
	actorcfg_llr->bound = rlcfg->inputbound;
	actorcfg_llr->cache = true;

	// Critic configuration
	criticcfg_llr->numsamp = critic_size;
	criticcfg_llr->numneighbours = critic_neigh;
	criticcfg_llr->indim = 2;
	criticcfg_llr->outdim = 1;
	criticcfg_llr->robust = 1;
	criticcfg_llr->measure = 2;
	criticcfg_llr->wrapping = wrapping;
	criticcfg_llr->weighting = weighting;
	criticcfg_llr->gamma = rlcfg->gamma;
	criticcfg_llr->alpha = critic_alpha;
	criticcfg_llr->lambda = 0.65;
	criticcfg_llr->bound = criticbound;
	criticcfg_llr->cache = true;

	// Process model configuration
	processcfg_llr->numsamp = process_size;
	processcfg_llr->numneighbours = process_neigh;
	processcfg_llr->indim = 3;
	processcfg_llr->outdim = 2;
	processcfg_llr->robust = 0;
	processcfg_llr->measure = 1;
	processcfg_llr->wrapping = nowrapping;
	processcfg_llr->weighting = proweighting;
	processcfg_llr->gamma = rlcfg->gamma;
	processcfg_llr->alpha = 0; // LLR process model does not require learning as samples are true values
	processcfg_llr->lambda = 0;
	processcfg_llr->bound = processbound;
	processcfg_llr->cache = true;

	// Create the LLR function approximator objects
	LocLinReg actor_llr(actorcfg_llr);
	LocLinReg critic_llr(criticcfg_llr);
	LocLinReg process_llr(processcfg_llr);

	time_t start, end;
	time(&start);

	// MLAC with LLR
	printf("Starting MLAC simulation using Local Linear Regression...\n");
	gsl_matrix * rewards = mlac(&invpend,&reward,actor_llr,critic_llr,process_llr,rlcfg);
	sprintf(outputname,"ML_%6.5f_%6.5f_%d_%d.dat",actorcfg_llr->alpha,criticcfg_llr->alpha,actorcfg_llr->numneighbours,criticcfg_llr->numneighbours);

	time(&end);

	FILE * outputfile = fopen(outputname,"wb");
	gsl_matrix_fwrite(outputfile,rewards);
	printf("File written to: %s\n",outputname);
	fclose(outputfile);

	printf("Total simulation time: %.0f seconds\n",difftime(end,start));

	// Free allocated memory
	gsl_matrix_free(criticbound);
	gsl_matrix_free(processbound);
	gsl_matrix_free(proweighting);
	gsl_matrix_free(Qrew);
	gsl_matrix_free(rewards);
	gsl_matrix_free(ubound);
	gsl_matrix_free(weighting);

	gsl_vector_free(nowrapping);
	gsl_vector_free(wrapping);
	gsl_vector_free(x0);

	delete rlcfg;
	delete actorcfg_llr;
	delete criticcfg_llr;
	delete processcfg_llr;

	return 0;
}

// Define the reward function for this RL learning experiment
double reward(const gsl_vector* x, const gsl_vector * u, const gsl_vector* xn) {
	double result;

	gsl_vector * y = gsl_vector_alloc(2);
	gsl_blas_dgemv(CblasNoTrans,1.,Qrew,x,0.,y);
	gsl_blas_ddot(x,y,&result);

	// Free allocated memory
	gsl_vector_free(y);

	return -(result + pow(gsl_vector_get(u,0),2));
}
