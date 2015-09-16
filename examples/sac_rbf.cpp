/*
 * examplerun.cpp
 *
 *  Created on: Nov 18, 2011
 *      Author: igrondman@tudelft.net
 */

#include<fstream>
#include<time.h>

#include "acrl_actorcritic.h"
#include "acrl_gaussrbf.h"
#include "acrl_rlearn.h"
#include "acrl_simmodels.h"

using namespace std;

double reward(const gsl_vector* x, const gsl_vector * u, const gsl_vector* xn);

// We only want the reward matrix to be initialised once
gsl_matrix * Qrew = gsl_matrix_calloc(2,2);

int main(int argc, char** argv) {
	double actor_alpha,critic_alpha, a_isect, c_isect, x1, x2;
	int r1,r2;
	char outputname[60];

	// General: set default learning rates
	actor_alpha = .08;
	critic_alpha = .3;

	// RBF: set default intersection heights of RBF's and number of RBF's
	a_isect = .6;
	c_isect = .6;

	// Set default sizes of state space
	x1 = M_PI;
	x2 = 8*M_PI;

	// RBF: set number of RBF's for each dimension (x1,x2,u)
	r1 = 15;
	r2 = 11;

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

	// General settings for the RL learning experiment
	gsl_vector * x0 = gsl_vector_alloc(2);
	gsl_vector_set(x0,0,M_PI);
	gsl_vector_set(x0,1,0);

	gsl_matrix * ubound = gsl_matrix_alloc(1,2);
	// Saturated input
	gsl_matrix_set(ubound,0,0,-3);
	gsl_matrix_set(ubound,0,1,3);

	gsl_vector * statewrap = gsl_vector_calloc(4);
	gsl_vector_set(statewrap,0,1);

	rlconfig * rlcfg = new rlconfig;
	rlcfg->Niterations = 20;
	rlcfg->Nepisodes = 600;
	rlcfg->gamma = 0.97;
	rlcfg->esigma = 1;
	rlcfg->erate = 3;
	rlcfg->episodelength = 3.;
	rlcfg->Ts = .03;
	rlcfg->odetime = rlcfg->Ts/10;
	rlcfg->x0 = x0;
	rlcfg->inputbound = ubound;
	rlcfg->statewrapping = statewrap;

	// Configure reward function
	gsl_matrix_set(Qrew,0,0,5.);
	gsl_matrix_set(Qrew,1,1,.1);

	gsl_matrix * ssdims = gsl_matrix_alloc(2,2);
	gsl_matrix_set(ssdims,0,0,-x1);
	gsl_matrix_set(ssdims,0,1,x1);
	gsl_matrix_set(ssdims,1,0,-x2);
	gsl_matrix_set(ssdims,1,1,x2);

	// Critic configuration
	gsl_matrix * criticbound = gsl_matrix_alloc(1,2);
	gsl_matrix_set(criticbound,0,0,-INFINITY);
	gsl_matrix_set(criticbound,0,1,INFINITY);

	// Radial basis functions
	rbfconfig * criticcfg_rbf = new rbfconfig;
	rbfconfig * actorcfg_rbf = new rbfconfig;

	gsl_vector * Nrbfs = gsl_vector_alloc(2);
	gsl_vector_set(Nrbfs,0,r1);
	gsl_vector_set(Nrbfs,1,r2);

	// Actor configuration
	actorcfg_rbf->ssdims = ssdims;
	actorcfg_rbf->outdim = 1;
	actorcfg_rbf->Nrbfs = Nrbfs;
	actorcfg_rbf->alpha = actor_alpha;
	actorcfg_rbf->bound = rlcfg->inputbound;
	actorcfg_rbf->gamma = rlcfg->gamma;
	actorcfg_rbf->lambda = 0; // 0.85;

	criticcfg_rbf->ssdims = ssdims;
	criticcfg_rbf->outdim = 1;
	criticcfg_rbf->Nrbfs = Nrbfs;
	criticcfg_rbf->alpha = critic_alpha;
	criticcfg_rbf->bound = criticbound;
	criticcfg_rbf->gamma = rlcfg->gamma;
	criticcfg_rbf->lambda = 0.65;

	RadialBasis actor_rbf(actorcfg_rbf,a_isect);
	RadialBasis critic_rbf(criticcfg_rbf,c_isect);

	time_t start, end;
	time(&start);

	// SAC with RBF
	gsl_matrix * rewards = sac(&invpend,&reward,actor_rbf,critic_rbf,rlcfg);
	sprintf(outputname,"SR_%6.5f_%6.5f_%4.2f_%4.2f_%d_%d.dat",actorcfg_rbf->alpha,criticcfg_rbf->alpha,a_isect,c_isect,r1,r2);

	time(&end);

	FILE * outputfile = fopen(outputname,"wb");
	gsl_matrix_fwrite(outputfile,rewards);
	printf("File written to: %s\n",outputname);
	fclose(outputfile);

	printf("Total simulation time: %.0f seconds\n",difftime(end,start));

	// Free allocated memory
	gsl_matrix_free(Qrew);
	gsl_matrix_free(ubound);
	gsl_matrix_free(ssdims);
	gsl_matrix_free(criticbound);

	gsl_vector_free(x0);
	gsl_vector_free(Nrbfs);
	gsl_vector_free(statewrap);


	delete rlcfg;

	delete criticcfg_rbf;
	delete actorcfg_rbf;

	gsl_matrix_free(rewards);

	return 0;
}

double reward(const gsl_vector* x, const gsl_vector * u, const gsl_vector* xn) {
	double result;

	gsl_vector * y = gsl_vector_alloc(2);
	gsl_blas_dgemv(CblasNoTrans,1.,Qrew,x,0.,y);
	gsl_blas_ddot(x,y,&result);

	// Free allocated memory
	gsl_vector_free(y);

	return -(result + pow(gsl_vector_get(u,0),2));
}
