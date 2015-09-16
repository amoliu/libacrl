/*
 * acrl_actorcritic_fh.cpp
 *
 *  Created on: Feb 4, 2013
 *      Author: grondman
 */

#include "acrl_actorcritic.h"

gsl_matrix * sac_fhvw(void (*eom)(gsl_vector*, gsl_vector*),double(*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*),double(*treward)(const gsl_vector*),FunApprox ** actor, FunApprox ** critic, rlconfig * rlcfg)
{
//	string acparam = "sac_acparams.dat";
//	string crparam = "sac_crparams.dat";
//	remove(acparam.c_str());
//	remove(crparam.c_str());

	// Calculate the number of steps to take in one full episode
	int Nsteps = round(rlcfg->episodelength/rlcfg->Ts);

	// Initialise rewards matrix that will be returned
	gsl_matrix * totalrewards = gsl_matrix_alloc(rlcfg->Niterations,rlcfg->Nepisodes);

	// Initialise a few variables used in the algorithm
	gsl_vector * xprev = gsl_vector_alloc(2), *x;
	gsl_vector * Vprev = gsl_vector_calloc(1);
	gsl_vector * V = gsl_vector_alloc(1);
	gsl_vector * action = gsl_vector_alloc(1);
	gsl_vector * u = gsl_vector_alloc(1);
	gsl_vector * odein = gsl_vector_alloc(3);
	gsl_vector * returnvec = gsl_vector_alloc(Nsteps);

	gsl_vector_view stateColumn, statePart, returnPart;

	double epreward, expl, r, TDe;

	// Generate the time span variable that will be used in simulations
	int odesteps = round(rlcfg->Ts/rlcfg->odetime)+1;
	gsl_vector * tspan = gsl_vector_alloc(odesteps);
	for (unsigned int k = 0; k < tspan->size; k++) {
		gsl_vector_set(tspan,k,k*rlcfg->odetime);
	}

	gsl_matrix * yout = gsl_matrix_alloc(3,odesteps);

	// Initialise a random number generator
	gsl_rng * rng = gsl_rng_alloc(gsl_rng_default);

	gsl_matrix * xu = gsl_matrix_alloc(3,Nsteps);

	// Loop through all iterations
	for (int i = 0; i < rlcfg->Niterations; i++) {
		printf("Learning curve %d of %d", i+1, rlcfg->Niterations);

		// Reset the actor and critic function approximators
		for (int rescount = 0; rescount < Nsteps; rescount++) {
			actor[rescount]->reset();
			critic[rescount]->reset();
		}

		// Loop through the episodes
		for (int j = 0; j < rlcfg->Nepisodes; j++) {
			// Reset eligibility traces
			for (int rescount = 0; rescount < Nsteps; rescount++) {
				actor[rescount]->resetEligibility();
				critic[rescount]->resetEligibility();
			}

			// Initialise the reward for this episode to zero
			epreward = 0;
			gsl_vector_set_zero(returnvec);

			// Set initial state
			gsl_vector_memcpy(xprev,rlcfg->x0);

			for (int k = 0; k < Nsteps; k++) {

				// Get action from actor
				actor[k]->getOutput(xprev,action);

				// Add exploration, saturate and recalculate exploration
				expl = 0;
				if (k % rlcfg->erate == 0 && j < rlcfg->Nepisodes-1) {
					expl = gsl_ran_gaussian(rng, rlcfg->esigma);
				}
				gsl_vector_memcpy(u,action);
				gsl_vector_add_constant(u,expl);
				sat(u,rlcfg->inputbound,u);

				// TODO The next line is causing trouble... WHY?!
//				 expl = gsl_vector_get(u,0) - gsl_vector_get(action,0);

				// Append the action to the state
				igsl_vector_append(xprev,u,odein);

				// Save the input u to vector of inputs in the last episode of the last iteration
//				if (i == rlcfg->Niterations-1 && j == rlcfg->Nepisodes-1) {
//					if (expl == 0)
						gsl_matrix_set_col(xu,k,odein);
//				}

				// Execute a time step in the system
				rk4_ti(eom, tspan, odein, yout);
				stateColumn = gsl_matrix_subcolumn(yout,odesteps-1,0,2);
				x = &stateColumn.vector;
				wrap(x,rlcfg->statewrapping);

				// Obtain the scalar reward
				r = reward(xprev,u,x);

				// Get value from critic
				critic[k]->getOutput(xprev,Vprev);

				// Calculate temporal difference error
				if (k == Nsteps-1)
					TDe = r + treward(x) - gsl_vector_get(Vprev,0);
				else {
					critic[k+1]->getOutput(x,V);
					TDe = r + gsl_vector_get(V,0) - gsl_vector_get(Vprev,0);
				}

				// Perform updates on the actor and critic
//				critic[k]->update(TDe,xprev,Vprev);
//				actor[k]->update(TDe*expl,xprev,u);

				// Add the reward of this time step to the total reward of the episode and update the return vector
				epreward += r;
				returnPart = gsl_vector_subvector(returnvec,0,k+1);
				gsl_vector_add_constant(&returnPart.vector,r);

				// Copy old values
				gsl_vector_memcpy(xprev,x);

			} // end of time step loop

			// Add terminal reward to the episode
			epreward += treward(x);
			gsl_vector_add_constant(returnvec,treward(x));

			for (int ui = 0; ui < Nsteps; ui++) {
				statePart = gsl_matrix_subcolumn(xu,ui,0,2);
				critic[ui]->getOutput(&statePart.vector,Vprev);
				TDe = gsl_vector_get(returnvec,ui) - gsl_vector_get(Vprev,0);
				critic[ui]->update(TDe,&statePart.vector,Vprev);
			}

			// Run an extra rollout to update the actors
			// Set initial state
			gsl_vector_memcpy(xprev,rlcfg->x0);

			for (int ai = 0; ai < Nsteps; ai++) {
				// Get action from actor
				actor[ai]->getOutput(xprev,action);

				// Add exploration, saturate and recalculate exploration
				expl = 0;
				if (ai % rlcfg->erate == 0) {
					expl = gsl_ran_gaussian(rng, rlcfg->esigma);
				}
				gsl_vector_memcpy(u,action);
				gsl_vector_add_constant(u,expl);
				sat(u,rlcfg->inputbound,u);

				// Append the action to the state
				igsl_vector_append(xprev,u,odein);

				// Execute a time step in the system
				rk4_ti(eom, tspan, odein, yout);
				stateColumn = gsl_matrix_subcolumn(yout,odesteps-1,0,2);
				x = &stateColumn.vector;
				wrap(x,rlcfg->statewrapping);

				// Obtain the scalar reward
				r = reward(xprev,u,x);

				// Get value from critic
				critic[ai]->getOutput(xprev,Vprev);

				// Calculate temporal difference error
				if (ai == Nsteps-1)
					TDe = r + treward(x) - gsl_vector_get(Vprev,0);
				else {
					critic[ai+1]->getOutput(x,V);
					TDe = r + gsl_vector_get(V,0) - gsl_vector_get(Vprev,0);
				}
				actor[ai]->update(TDe*expl,xprev,u);
			}

			// Add reward for this episode to the rewards vector
			gsl_matrix_set(totalrewards,i,j,epreward);

//			if (epreward > epold) {
//				actor.writeParam(acparam);
//				critic.writeParam(crparam);
//				printf("File written to: %s at episode %d (total reward = %8.f)\n",param_out.c_str(),j,epreward);
//				epold = epreward;
//			}

		} // End of episode loop

		printf(": first episode = %f, last episode = %f\n",gsl_matrix_get(totalrewards,i,0),gsl_matrix_get(totalrewards,i,rlcfg->Nepisodes-1));
		//printf(": first episode = %f, last episode = %f\n",testDouble(),testDouble());

		if (i == rlcfg->Niterations-1) {
			FILE * outputfile = fopen("stateinputs_SAC.dat","wb");
			gsl_matrix_fwrite(outputfile,xu);
			fclose(outputfile);
		}

	} // End of iteration loop

	// Free memory allocated earlier
	gsl_matrix_free(yout);
	gsl_matrix_free(xu);

	gsl_vector_free(xprev);
	gsl_vector_free(Vprev);
	gsl_vector_free(V);
	gsl_vector_free(action);
	gsl_vector_free(u);
	gsl_vector_free(odein);
	gsl_vector_free(tspan);

	gsl_rng_free(rng);

	return totalrewards;
}

gsl_matrix *mlac_fhvw(void(*eom)(gsl_vector*, gsl_vector*), double (*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*), double(*treward)(const gsl_vector*), FunApprox ** actor, FunApprox ** critic, FunApprox & process, rlconfig *rlcfg)
{
	// Initialise rewards matrix that will be returned
	gsl_matrix * totalrewards = gsl_matrix_alloc(rlcfg->Niterations,rlcfg->Nepisodes);

	// Initialise a few variables used in the algorithm
	gsl_vector * xprev = gsl_vector_alloc(2), *x;
	gsl_vector * xunwrapped = gsl_vector_alloc(2);
	gsl_vector * Vprev = gsl_vector_calloc(1);
	gsl_vector * V = gsl_vector_alloc(1);
	gsl_vector * action = gsl_vector_alloc(1);
	gsl_vector * u = gsl_vector_alloc(1);
	gsl_vector * u_policy = gsl_vector_alloc(1);
	gsl_vector * odein = gsl_vector_alloc(3);
	gsl_vector * ode_policy = gsl_vector_alloc(3);
	gsl_vector * processout = gsl_vector_alloc(2);
	gsl_vector * pmoutpolicy = gsl_vector_alloc(2);
	gsl_vector * Dprocess = gsl_vector_calloc(2);

	gsl_vector_view stateColumn, processModelRow, criticColumn;

	gsl_matrix * beta_c = gsl_matrix_alloc(2,1);
	gsl_matrix * beta_cn = gsl_matrix_alloc(2,1);
	gsl_matrix * beta_p = gsl_matrix_alloc(3,2);
	gsl_matrix * beta_pn = gsl_matrix_alloc(3,2);

	double epreward, expl, dVdu, r, TDe;

	// Generate the time span variable that will be used in simulations
	int odesteps = round(rlcfg->Ts/rlcfg->odetime)+1;
	gsl_vector * tspan = gsl_vector_alloc(odesteps);
	for (unsigned int k = 0; k < tspan->size; k++) {
		gsl_vector_set(tspan,k,k*rlcfg->odetime);
	}

	gsl_matrix * yout = gsl_matrix_alloc(3,odesteps);

	// Initialise a random number generator
	gsl_rng * rng = gsl_rng_alloc(gsl_rng_default);

	// Calculate the number of steps to take in one full episode
	int Nsteps = round(rlcfg->episodelength/rlcfg->Ts);

	gsl_matrix * xu = gsl_matrix_alloc(3,Nsteps);

	// Loop through all iterations
	for (int i = 0; i < rlcfg->Niterations; i++) {
		printf("Learning curve %d of %d", i+1, rlcfg->Niterations);

		// Reset all function approximators
		for (int rescount = 0; rescount < Nsteps; rescount++) {
			actor[rescount]->reset();
			critic[rescount]->reset();
		}
		process.reset();


		// Loop through the episodes
		for (int j = 0; j < rlcfg->Nepisodes; j++) {
			// Reset eligibility traces
			for (int rescount = 0; rescount < Nsteps; rescount++) {
				critic[rescount]->resetEligibility();
				actor[rescount]->resetEligibility();
			}

			// Initialise the reward for this episode to zero
			epreward = 0;

			// Set initial state
			gsl_vector_memcpy(xprev,rlcfg->x0);

			// Get value of initial state
			critic[0]->getOutput(xprev,Vprev,beta_c);

			for (int k = 0; k < Nsteps; k++) {
				// Get action from actor
				actor[k]->getOutput(xprev,u_policy);

				// Add exploration and saturate
				expl = 0;
				if (k % rlcfg->erate == 0 && j < rlcfg->Nepisodes-1)
				{
					expl = gsl_ran_gaussian(rng, rlcfg->esigma);
				}
				gsl_vector_memcpy(u,u_policy);
				gsl_vector_add_constant(u,expl);
				sat(u,rlcfg->inputbound,u);
				sat(u_policy,rlcfg->inputbound,u_policy);

				// Execute a time step in the system
				igsl_vector_append(xprev,u_policy,ode_policy);
				igsl_vector_append(xprev,u,odein);

				// Save the input u to vector of inputs in the last episode of the last iteration
				if (i == rlcfg->Niterations-1 && j == rlcfg->Nepisodes-1) {
					if (expl == 0)
						gsl_matrix_set_col(xu,k,odein);
				}

				// Execute a time step in the system
				rk4_ti(eom, tspan, odein, yout);
				stateColumn = gsl_matrix_subcolumn(yout,odesteps-1,0,2);
				x = &stateColumn.vector;
				gsl_vector_memcpy(xunwrapped,x);
				wrap(x,rlcfg->statewrapping);

				// Obtain the scalar reward
				r = reward(xprev,u,x);

				// Get the process model
				process.getOutput(ode_policy,pmoutpolicy,beta_pn);
				process.getOutput(odein,processout,beta_p);
//
				// Setting the df/du to zero here for RBF use // TODO: Make this configurable
				if (fabs(gsl_vector_get(u,0)) > 2.75) {
					processModelRow = gsl_matrix_row(beta_p,beta_p->size1-1);
					gsl_vector_set_all(&processModelRow.vector,0);
				}
				if (fabs(gsl_vector_get(u_policy,0)) > 2.75) {
					processModelRow = gsl_matrix_row(beta_pn,beta_pn->size1-1);
					gsl_vector_set_all(&processModelRow.vector,0);
				}
				// End of test code

				if (k == Nsteps-1) {
					gsl_matrix_set(beta_cn,0,0,-10*gsl_vector_get(x,0));
					gsl_matrix_set(beta_cn,1,0,-.2*gsl_vector_get(x,1));
				}
				else {
					critic[k+1]->getOutput(pmoutpolicy,V,beta_cn);
				}

				// Calculate the gradient dV/du for the actor update
				processModelRow = gsl_matrix_row(beta_pn,beta_pn->size1-1);
				criticColumn = gsl_matrix_column(beta_cn,0);
				gsl_blas_ddot(&criticColumn.vector,&processModelRow.vector,&dVdu);

				// Update the process model
				gsl_vector_memcpy(Dprocess,xunwrapped);
				gsl_vector_sub(Dprocess,processout);
				process.update(Dprocess,odein,xunwrapped);
//				process.update(Dprocess,ode_policy,xunwrapped);

				// Update the actor
				dVdu += -2*gsl_vector_get(u_policy,0);
				if (j < rlcfg->Nepisodes-1) {
					actor[k]->update(dVdu,xprev,u);
				}

				// Get value from critic
				critic[k]->getOutput(xprev,Vprev,beta_c);

				// Calculate temporal difference error
				if (k == Nsteps-1)
					TDe = r + treward(x) - gsl_vector_get(Vprev,0);
				else {
					critic[k+1]->getOutput(x,V);
					TDe = r + gsl_vector_get(V,0) - gsl_vector_get(Vprev,0);
				}

				// Perform update on the critic
				critic[k]->update(TDe,xprev,Vprev);

				// Add the reward of this time step to the total reward of the episode
				epreward += r;

				// Copy old values
				gsl_vector_memcpy(xprev,x);

			} // end of time step loop

			// Add terminal reward to the episode
			epreward += treward(x);

			// Add reward for this episode to the rewards vector
			gsl_matrix_set(totalrewards,i,j,epreward);

//			actor.writeParam("actorMLACparam.dat");
//			critic.writeParam("criticMLACparam.dat");

		} // End of episode loop

		printf(": first episode = %f, last episode = %f\n",gsl_matrix_get(totalrewards,i,0),gsl_matrix_get(totalrewards,i,rlcfg->Nepisodes-1));
		//printf(": first episode = %f, last episode = %f\n",testDouble(),testDouble());
		if (i == rlcfg->Niterations-1) {
			FILE * outputfile = fopen("stateinputs_MLAC_FH","wb");
			gsl_matrix_fwrite(outputfile,xu);
			fclose(outputfile);
		}
	} // End of iteration loop

	// Free memory allocated earlier
	gsl_matrix_free(yout);
	gsl_matrix_free(beta_c);
	gsl_matrix_free(beta_cn);
	gsl_matrix_free(beta_p);
	gsl_matrix_free(beta_pn);

	gsl_vector_free(xprev);
	gsl_vector_free(xunwrapped);
	gsl_vector_free(Vprev);
	gsl_vector_free(V);
	gsl_vector_free(action);
	gsl_vector_free(u);
	gsl_vector_free(odein);
	gsl_vector_free(tspan);
	gsl_vector_free(processout);
	gsl_vector_free(pmoutpolicy);
	gsl_matrix_free(xu);

	gsl_vector_free(Dprocess);
	gsl_vector_free(u_policy);
	gsl_vector_free(ode_policy);

	gsl_rng_free(rng);

	return totalrewards;
}
