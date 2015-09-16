/*
 * acrl_actorcritic_fh.cpp
 *
 *  Created on: Feb 4, 2013
 *      Author: grondman
 */

#include "acrl_actorcritic.h"

gsl_matrix * sac_fh(void (*eom)(gsl_vector*, gsl_vector*),double(*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*),double(*treward)(const gsl_vector*),FunApprox & actor, FunApprox & critic, rlconfig * rlcfg)
{
//	string acparam = "sac_acparams.dat";
//	string crparam = "sac_crparams.dat";
//	remove(acparam.c_str());
//	remove(crparam.c_str());

	// Initialise rewards matrix that will be returned
	gsl_matrix * totalrewards = gsl_matrix_alloc(rlcfg->Niterations,rlcfg->Nepisodes);

	// Initialise a few variables used in the algorithm
	gsl_vector * xprev = gsl_vector_alloc(3), *x;
	gsl_vector * Vprev = gsl_vector_calloc(1);
	gsl_vector * V = gsl_vector_alloc(1);
	gsl_vector * action = gsl_vector_alloc(1);
	gsl_vector * u = gsl_vector_alloc(1);
	gsl_vector * odein = gsl_vector_alloc(3);

	gsl_vector_view stateColumn;

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

	// Calculate the number of steps to take in one full episode
	int Nsteps = round(rlcfg->episodelength/rlcfg->Ts);

	gsl_matrix * xu = gsl_matrix_alloc(3,Nsteps);

	// Loop through all iterations
	for (int i = 0; i < rlcfg->Niterations; i++) {
		printf("Learning curve %d of %d", i+1, rlcfg->Niterations);

		// Reset the actor and critic function approximators
		actor.reset();
		critic.reset();

		// Loop through the episodes
		for (int j = 0; j < rlcfg->Nepisodes; j++) {
			// Reset eligibility traces
			critic.resetEligibility();
			actor.resetEligibility();

			// Initialise the reward for this episode to zero
			epreward = 0;

			// Set initial state
			gsl_vector_memcpy(xprev,rlcfg->x0);

			// Get value of initial state
			critic.getOutput(xprev,Vprev);

			for (int k = 0; k < Nsteps; k++) {

				// Get action from actor
				actor.getOutput(xprev,action);

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
				gsl_vector_memcpy(odein,xprev);
				gsl_vector_set(odein,2,gsl_vector_get(u,0));

				// Save the input u to vector of inputs in the last episode of the last iteration
				if (i == rlcfg->Niterations-1 && j == rlcfg->Nepisodes-1) {
					if (expl == 0)
						gsl_matrix_set_col(xu,k,odein);
				}

				// Execute a time step in the system
				rk4_ti(eom, tspan, odein, yout);
				stateColumn = gsl_matrix_column(yout,odesteps-1);
				x = &stateColumn.vector;
				wrap(x,rlcfg->statewrapping);
				gsl_vector_set(x,2,k+1);

				// Obtain the scalar reward
				r = reward(xprev,u,x);

				// Get value from critic
				critic.getOutput(xprev,Vprev);

				// Calculate temporal difference error
				if (k == Nsteps-1)
					TDe = r + treward(x) - gsl_vector_get(Vprev,0);
				else {
					critic.getOutput(x,V);
					TDe = r + gsl_vector_get(V,0) - gsl_vector_get(Vprev,0);
				}

				// Perform updates on the actor and critic
				critic.update(TDe,xprev,Vprev);
				actor.update(TDe*expl,xprev,u);

				// Add the reward of this time step to the total reward of the episode
				epreward += r;

				// Copy old values
				gsl_vector_memcpy(xprev,x);

			} // end of time step loop

			// Add terminal reward to the episode
			epreward += treward(x);

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
			FILE * outputfile = fopen("stateinputs_SAC_FH.dat","wb");
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

gsl_matrix *mlac_fh(void(*eom)(gsl_vector*, gsl_vector*), double (*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*), double(*treward)(const gsl_vector*), FunApprox & actor, FunApprox & critic, FunApprox & process, rlconfig *rlcfg)
{
	// Initialise rewards matrix that will be returned
	gsl_matrix * totalrewards = gsl_matrix_alloc(rlcfg->Niterations,rlcfg->Nepisodes);

	// Initialise a few variables used in the algorithm
	gsl_vector * xprev = gsl_vector_alloc(3), *x;
	gsl_vector * xunwrapped = gsl_vector_alloc(2);
	gsl_vector * Vprev = gsl_vector_calloc(1);
	gsl_vector * V = gsl_vector_alloc(1);
	gsl_vector * action = gsl_vector_alloc(1);
	gsl_vector * u = gsl_vector_alloc(1);
	gsl_vector * u_policy = gsl_vector_alloc(1);
	gsl_vector * odein = gsl_vector_alloc(3);
	gsl_vector * ode_policy = gsl_vector_alloc(3);
	gsl_vector * processout = gsl_vector_alloc(2);
	gsl_vector * pmoutpolicy = gsl_vector_alloc(3);
	gsl_vector * Dprocess = gsl_vector_calloc(2);

	gsl_vector_view stateColumn, statePart, processOut, processModelRow, criticColumn;

	gsl_matrix * beta_c = gsl_matrix_alloc(3,1);
	gsl_matrix * beta_cn = gsl_matrix_alloc(3,1);
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
		actor.reset();
		critic.reset();
		process.reset();

		// Loop through the episodes
		for (int j = 0; j < rlcfg->Nepisodes; j++) {
			// Reset eligibility traces
			critic.resetEligibility();
			actor.resetEligibility();

			// Initialise the reward for this episode to zero
			epreward = 0;

			// Set initial state
			gsl_vector_memcpy(xprev,rlcfg->x0);

			// Get value of initial state
			critic.getOutput(xprev,Vprev,beta_c);

			// Get action from actor
			actor.getOutput(xprev,u_policy);

			for (int k = 0; k < Nsteps; k++) {
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

				// Append the action to the state
				gsl_vector_memcpy(ode_policy,xprev);
				gsl_vector_set(ode_policy,2,gsl_vector_get(u_policy,0));
				gsl_vector_memcpy(odein,xprev);
				gsl_vector_set(odein,2,gsl_vector_get(u,0));

				// Save the input u to vector of inputs in the last episode of the last iteration
				if (i == rlcfg->Niterations-1 && j == rlcfg->Nepisodes-1) {
					if (expl == 0)
						gsl_matrix_set_col(xu,k,odein);
				}

				// Execute a time step in the system
				rk4_ti(eom, tspan, odein, yout);
				stateColumn = gsl_matrix_column(yout,odesteps-1);
				x = &stateColumn.vector;
				statePart = gsl_vector_subvector(x,0,2);
				gsl_vector_memcpy(xunwrapped,&statePart.vector);
				wrap(x,rlcfg->statewrapping);
				gsl_vector_set(x,2,k+1);

				// Obtain the scalar reward
				r = reward(xprev,u,x);

				// Get the process model
				processOut = gsl_vector_subvector(pmoutpolicy,0,2);
				process.getOutput(ode_policy,&processOut.vector,beta_pn);
				gsl_vector_set(pmoutpolicy,2,k+1);
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

				// In the last time step, the exact value of beta_cn is known because of the
				// terminal reward.
				if (k == Nsteps -1) {
					gsl_matrix_set(beta_cn,0,0,-10*gsl_vector_get(x,0));
					gsl_matrix_set(beta_cn,1,0,-.2*gsl_vector_get(x,1));
					gsl_matrix_set(beta_cn,2,0,0);
				}
				else {
					critic.getOutput(pmoutpolicy,V,beta_cn);
				}

				// Calculate the gradient dV/du for the actor update
				criticColumn = gsl_matrix_subcolumn(beta_cn,0,0,2);
				processModelRow = gsl_matrix_row(beta_pn,beta_pn->size1-1);
				gsl_blas_ddot(&criticColumn.vector,&processModelRow.vector,&dVdu);

				// Update the process model
				gsl_vector_memcpy(Dprocess,xunwrapped);
				gsl_vector_sub(Dprocess,processout);
				process.update(Dprocess,odein,xunwrapped);
//				process.update(Dprocess,ode_policy,xunwrapped);

				// Update the actor
				dVdu += -2*gsl_vector_get(u_policy,0);
				if (j < rlcfg->Nepisodes-1) {
					actor.update(dVdu,xprev,u);
				}

				// Get action from actor
				actor.getOutput(x,u_policy);

				// Get value from critic
				critic.getOutput(xprev,Vprev,beta_c);
				critic.getOutput(x,V,beta_c);

				// Calculate temporal difference error
				if (k == Nsteps-1)
					TDe = r + treward(x) - gsl_vector_get(Vprev,0);
				else {
					critic.getOutput(x,V);
					TDe = r + gsl_vector_get(V,0) - gsl_vector_get(Vprev,0);
				}

				// Perform update on the critic
				critic.update(TDe,xprev,Vprev);

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

gsl_matrix *rmac_fh(void(*eom)(gsl_vector*, gsl_vector*), double (*reward)(const gsl_vector*, const gsl_vector*, const gsl_vector*), double(*treward)(const gsl_vector*), FunApprox & critic, FunApprox & process, FunApprox & reference, rlconfig *rlcfg)
{
	// Initialise rewards matrix that will be returned
	gsl_matrix * totalrewards = gsl_matrix_alloc(rlcfg->Niterations,rlcfg->Nepisodes);

	// Initialise a few variables used in the algorithm
	gsl_vector * xprev = gsl_vector_alloc(3), *x;
	gsl_vector * xunwrapped = gsl_vector_alloc(2);
	gsl_vector * Vprev = gsl_vector_calloc(1);
	gsl_vector * Ref = gsl_vector_alloc(2);
	gsl_vector * Refprev = gsl_vector_alloc(2);
	gsl_vector * Refnew = gsl_vector_alloc(2);
	gsl_vector * DRef = gsl_vector_alloc(2);
	gsl_vector * V = gsl_vector_alloc(1);
	gsl_vector * u = gsl_vector_alloc(1);
	gsl_vector * odein = gsl_vector_alloc(3);
	gsl_matrix * xplus = gsl_matrix_alloc(2,3);
	gsl_vector * V1 = gsl_vector_alloc(1);
	gsl_vector * V2 = gsl_vector_alloc(1);
	gsl_vector * xaug = gsl_vector_alloc(3);
	gsl_vector * processout = gsl_vector_alloc(2);
	gsl_vector * Dprocess = gsl_vector_calloc(2);

	gsl_vector_view stateColumn, statePart, inputPart, processOut;
	gsl_vector_view nextStateRow;

	gsl_matrix * beta_p = gsl_matrix_alloc(3,2);

	gsl_matrix * Refsat = gsl_matrix_alloc(3,2);

	char xufilename[60];

	gsl_matrix_set(Refsat,0,0,-M_PI);
	gsl_matrix_set(Refsat,0,1,M_PI);
	gsl_matrix_set(Refsat,1,0,-8*M_PI);
	gsl_matrix_set(Refsat,1,1,8*M_PI);
	gsl_matrix_set(Refsat,2,0,-INFINITY);
	gsl_matrix_set(Refsat,2,1,INFINITY);
//	gsl_matrix_scale(Refsat,1.1);


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

	// Calculate the number of steps to take in one full episode
	int Nsteps = round(rlcfg->episodelength/rlcfg->Ts);

	gsl_matrix * xu = gsl_matrix_alloc(3,Nsteps);

	// Loop through all iterations
	for (int i = 0; i < rlcfg->Niterations; i++) {
		printf("Learning curve %d of %d", i+1, rlcfg->Niterations);

		// Reset the actor and critic function approximators
		critic.reset();
		process.reset();
		reference.reset();

		// Loop through the episodes
		for (int j = 0; j < rlcfg->Nepisodes; j++) {
			// Reset eligibility traces
			critic.resetEligibility();

			// Initialise the reward for this episode to zero
			epreward = 0;

			// Set initial state
			gsl_vector_memcpy(xprev,rlcfg->x0);

			// Get value of initial state
			reference.getOutput(xprev,Refprev);

			// Calculate input from reference/process model
			gsl_vector_memcpy(odein,xprev);
			statePart = gsl_vector_subvector(xprev,0,2);
			process.getAction(&statePart.vector,Refprev,odein);
			inputPart = gsl_vector_subvector(odein,odein->size-1,u->size);
			gsl_vector_memcpy(u,&inputPart.vector);
			expl = 0;
			if (j == 0) // Explore on first episode
			{
				// expl = .25;
				expl = gsl_ran_gaussian(rng, rlcfg->esigma);
			}
			gsl_vector_add_constant(u,expl);
			sat(u,rlcfg->inputbound,u);

			for (int k = 0; k < Nsteps; k++) {
				// Execute a time step in the system
				gsl_vector_memcpy(odein,xprev);
				gsl_vector_set(odein,2,gsl_vector_get(u,0));
				rk4_ti(eom, tspan, odein, yout);
				stateColumn = gsl_matrix_column(yout,odesteps-1);
				x = &stateColumn.vector;
				statePart = gsl_vector_subvector(x,0,2);
				gsl_vector_memcpy(xunwrapped,&statePart.vector);
				wrap(x,rlcfg->statewrapping);
				gsl_vector_set(x,2,k+1);

				// Save the input u to vector of inputs in the last episode of the last iteration
//				if (i == rlcfg->Niterations-1 && j == rlcfg->Nepisodes-1) {
				if (j == rlcfg->Nepisodes-1) {
						gsl_matrix_set_col(xu,k,odein);
				}

				// Obtain the scalar reward
				r = reward(xprev,u,x);

				// Get the process model
				process.getOutput(odein,processout,beta_p);
//				process.getOutput(ode_policy,processout,beta_p); // TODO do we need unsaturated input here?

				// Calculate desired next state / update reference memory
				gsl_vector_memcpy(xaug,odein);
				gsl_vector_set(xaug,2,-3);
				nextStateRow = gsl_matrix_subrow(xplus,0,0,2);
				process.getOutput(xaug,&nextStateRow.vector);
				gsl_vector_set(xaug,2,3);
				nextStateRow = gsl_matrix_subrow(xplus,1,0,2);
				process.getOutput(xaug,&nextStateRow.vector);
				gsl_matrix_set(xplus,0,2,k+1);
				gsl_matrix_set(xplus,1,2,k+1);
				// TODO Wrap the possible outputs?
				// wrap(xplus);

				// Calculate the values for both possible desired states
				nextStateRow = gsl_matrix_row(xplus,0);
				critic.getOutput(&nextStateRow.vector,V1);
				nextStateRow = gsl_matrix_row(xplus,1);
				critic.getOutput(&nextStateRow.vector,V2);

				// Decide on the correct desired state based on value
				if (gsl_vector_get(V1,0) >= gsl_vector_get(V2,0)) {
					nextStateRow = gsl_matrix_subrow(xplus,0,0,2);
					gsl_vector_memcpy(Refnew,&nextStateRow.vector);
				}
				else {
					nextStateRow = gsl_matrix_subrow(xplus,1,0,2);
					gsl_vector_memcpy(Refnew,&nextStateRow.vector);
				}
				// Saturate Refnew so that reference can not be too far outside of the (known) state space
//				sat(Refnew,Refsat,Refnew);

				// Get reference state for current state
				reference.getOutput(x,Ref);

				// Determine new action
				// Calculate input from process / reference model
				gsl_vector * odecopy = gsl_vector_alloc(odein->size);
				gsl_vector_memcpy(odecopy,odein);
				statePart = gsl_vector_subvector(x,0,2);
				process.getAction(&statePart.vector,Ref,odein);
				inputPart = gsl_vector_subvector(odein,x->size-1,u->size);
				gsl_vector_memcpy(u,&inputPart.vector);
				expl = 0;
				if (k % rlcfg->erate == 0 && j < rlcfg->Nepisodes-1) {
					// expl = .25; // Debugging purposes
					expl = gsl_ran_gaussian(rng, rlcfg->esigma);
				}
				gsl_vector_add_constant(u,expl);
				sat(u,rlcfg->inputbound,u);

				// Don't learn in the final episode
				if (j < rlcfg->Nepisodes-1)
				{
					// Learn process model
					gsl_vector_memcpy(Dprocess,xunwrapped);
					gsl_vector_sub(Dprocess,processout);
					process.update(Dprocess,odecopy,xunwrapped);
					gsl_vector_free(odecopy);

					// Perform update critic
					// Get value for previous and current state
					critic.getOutput(xprev,Vprev);
					// Calculate temporal difference error
					if (k == Nsteps-1)
						TDe = r + treward(x) - gsl_vector_get(Vprev,0);
					else {
						critic.getOutput(x,V);
						TDe = r + gsl_vector_get(V,0) - gsl_vector_get(Vprev,0);
					}
					critic.update(TDe,xprev,Vprev);

					// Update reference model
					gsl_vector_memcpy(DRef,Refnew);
					gsl_vector_sub(DRef,Refprev);
					reference.update(DRef,xprev,Refnew);
				}

				// Add the reward of this time step to the total reward of the episode
				epreward += r;

				// Copy old values
				gsl_vector_memcpy(xprev,x);
				gsl_vector_memcpy(Refprev,Ref);

			} // end of time step loop

			critic.writeParam("critic_RMACLLR.dat");
			reference.writeParam("reference_RMACLLR.dat");

			// Add reward for this episode to the rewards vector
			gsl_matrix_set(totalrewards,i,j,epreward);

		} // End of episode loop

		printf(": first episode = %f, last episode = %f\n",gsl_matrix_get(totalrewards,i,0),gsl_matrix_get(totalrewards,i,rlcfg->Nepisodes-1));
		//printf(": first episode = %f, last episode = %f\n",testDouble(),testDouble());

//		if (i == rlcfg->Niterations-1) {
			sprintf(xufilename,"stateinputs_RMAC_%02d_FH",i+1);
			FILE * outputfile = fopen(xufilename,"wb");
			gsl_matrix_fwrite(outputfile,xu);
			fclose(outputfile);
//		}

	} // End of iteration loop

	// Free memory allocated earlier
	gsl_matrix_free(yout);
	gsl_matrix_free(beta_p);
	gsl_matrix_free(xplus);
	gsl_matrix_free(Refsat);
	gsl_matrix_free(xu);

	gsl_vector_free(xprev);
	gsl_vector_free(xunwrapped);
	gsl_vector_free(Vprev);
	gsl_vector_free(Ref);
	gsl_vector_free(Refprev);
	gsl_vector_free(Refnew);
	gsl_vector_free(DRef);
	gsl_vector_free(V);
	gsl_vector_free(u);
	gsl_vector_free(odein);
	gsl_vector_free(tspan);
	gsl_vector_free(V1);
	gsl_vector_free(V2);
	gsl_vector_free(xaug);
	gsl_vector_free(processout);
	gsl_vector_free(Dprocess);

	gsl_rng_free(rng);

	return totalrewards;
}
