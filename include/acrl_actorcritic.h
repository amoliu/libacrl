/*
 * actorcritic.h
 *
 *  Created on: Nov 18, 2011
 *      Author: igrondman@tudelft.net
 */

#ifndef ACRL_ACTORCRITIC_H_
#define ACRL_ACTORCRITIC_H_

#include<algorithm>
#include<string>
#include<vector>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>

#include "acrl_fapprox.h"
#include "acrl_misc.h"
#include "acrl_rlearn.h"
#include "acrl_solvers.h"

using namespace std;

// Actor-critic methods (infinite horizon)
gsl_matrix * sac(void (*eom)(gsl_vector*, gsl_vector*),double(*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*),FunApprox & actor, FunApprox & critic, rlconfig * rlcfg);

gsl_matrix * mlac(void (*eom)(gsl_vector*, gsl_vector*),double(*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*),FunApprox & actor, FunApprox & critic, FunApprox & process, rlconfig * rlcfg);

gsl_matrix * rmac(void (*eom)(gsl_vector*, gsl_vector*),double(*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*),FunApprox & critic, FunApprox & process, FunApprox & reference, rlconfig * rlcfg);

// Actor-critic methods (finite horizon)
gsl_matrix * sac_fh(void (*eom)(gsl_vector*, gsl_vector*),double(*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*),double(*termreward)(const gsl_vector*),FunApprox & actor, FunApprox & critic, rlconfig * rlcfg);
gsl_matrix * mlac_fh(void(*eom)(gsl_vector*, gsl_vector*), double (*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*), double(*treward)(const gsl_vector*), FunApprox & actor, FunApprox & critic, FunApprox & process, rlconfig *rlcfg);
gsl_matrix * rmac_fh(void(*eom)(gsl_vector*, gsl_vector*), double (*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*), double(*treward)(const gsl_vector*), FunApprox & critic, FunApprox & process, FunApprox & reference, rlconfig *rlcfg);
gsl_matrix * sac_fhvw(void (*eom)(gsl_vector*, gsl_vector*),double(*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*),double(*termreward)(const gsl_vector*),FunApprox ** actor, FunApprox ** critic, rlconfig * rlcfg);
gsl_matrix * mlac_fhvw(void(*eom)(gsl_vector*, gsl_vector*), double (*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*), double(*treward)(const gsl_vector*),FunApprox ** actor, FunApprox ** critic, FunApprox & process, rlconfig *rlcfg);

// Actor-critic methods with delta penalty's (infinite horizon)
gsl_matrix * rmac_delta(void (*eom)(gsl_vector*, gsl_vector*),double(*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*),FunApprox & critic, FunApprox & process, FunApprox & reference, rlconfig * rlcfg);

gsl_matrix * mlac_delta(void (*eom)(gsl_vector*, gsl_vector*),double(*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*),FunApprox & actor, FunApprox & critic, FunApprox & process, rlconfig * rlcfg);

#endif /* ACRL_ACTORCRITIC_H_ */
