/*
 * acrl_twolink.cpp
 *
 *  Created on: Oct 2, 2013
 *      Author: igrondman
 */

#include "acrl_simmodels.h"

// 2-link pendulum
void twolink(gsl_vector *in, gsl_vector *out)
{
	double q_i1, q_i2, qd_i1, qd_i2, u1, u2;

	// Give trivial names to elements of input
	q_i1 = gsl_vector_get(in,0);
	q_i2 = gsl_vector_get(in,1);
	qd_i1 = gsl_vector_get(in,2);
	qd_i2 = gsl_vector_get(in,3);
	u1 = gsl_vector_get(in,4);
	u2 = gsl_vector_get(in,5);

	// Derivative of first two elements equals the third and fourth element
	gsl_vector_set(out,0,qd_i1);
	gsl_vector_set(out,1,qd_i2);

	// The double-dot equations
	gsl_vector_set(out,2,(2285015473062*sin(q_i1 - q_i2)*pow(qd_i2,2) - 28618840000000*u1 + 107392803819408*sin(q_i1) + cos(q_i1 - q_i2)*(849988351107*sin(q_i1 - q_i2)*pow(qd_i1,2) + 10645740000000*u2 - 21799701240156*sin(q_i2)))/(849988351107*pow(cos(q_i1 - q_i2),2) - 7827739975751));
	gsl_vector_set(out,3,-((5823582267447*sin(q_i1 - q_i2)*pow(qd_i1,2))/2 + (109406810000000*u2)/3 - 74678878488438*sin(q_i2) + cos(q_i1 - q_i2)*(849988351107*sin(q_i1 - q_i2)*pow(qd_i2,2) - 10645740000000*u1 + 39948365039688*sin(q_i1)))/(849988351107*pow(cos(q_i1 - q_i2),2) - 7827739975751));

	// Set derivative of inputs to zero
	gsl_vector_set(out,4,0);
	gsl_vector_set(out,5,0);
}

void twolinksym(gsl_vector *in, gsl_vector *out)
{
	double q_i1, q_i2, qd_i1, qd_i2, u1, u2;

	double m_cm1 = 0.809;
	double m_cm2 = 0.852;

	double I1 = 0.1449;
	double I2 = 0.1635;

	double g = 0;// 9.81;

	double l1 = 0.3825;
	double lComLink1 = 0.0700;
	double lComLink2 = 0.2450;

	// Give trivial names to elements of input
	q_i1 = gsl_vector_get(in,0);
	q_i2 = gsl_vector_get(in,1);
	qd_i1 = gsl_vector_get(in,2);
	qd_i2 = gsl_vector_get(in,3);
	u1 = gsl_vector_get(in,4);
	u2 = gsl_vector_get(in,5);

	// Derivative of first two elements equals the third and fourth element
	gsl_vector_set(out,0,qd_i1);
	gsl_vector_set(out,1,qd_i2);

	// The double-dot equations
	double sinq1mq2 = sin(q_i1 - q_i2);
	double cosq1mq2 = cos(q_i1 - q_i2);
	gsl_vector_set(out,2,-(pow(l1*lComLink2*m_cm2*qd_i1,2)*sin(2*q_i1 - 2*q_i2) - 2*pow(lComLink2,2)*m_cm2*u1 - 2*I2*u1 + 2*l1*lComLink2*m_cm2*u2*cosq1mq2 + 2*l1*pow(lComLink2,3)*pow(m_cm2*qd_i2,2)*sinq1mq2 + g*l1*pow(lComLink2*m_cm2,2)*sin(q_i1) + 2*I2*g*l1*m_cm2*sin(q_i1) + 2*I2*g*lComLink1*m_cm1*sin(q_i1) + g*l1*pow(lComLink2*m_cm2,2)*sin(q_i1 - 2*q_i2) + 2*g*lComLink1*pow(lComLink2,2)*m_cm1*m_cm2*sin(q_i1) + 2*I2*l1*lComLink2*m_cm2*pow(qd_i2,2)*sinq1mq2)/(2*I1*I2 + pow(l1*lComLink2*m_cm2,2) + 2*I2*pow(l1,2)*m_cm2 + 2*I2*pow(lComLink1,2)*m_cm1 + 2*I1*pow(lComLink2,2)*m_cm2 - pow(l1*lComLink2*m_cm2,2)*cos(2*q_i1 - 2*q_i2) + 2*pow(lComLink1*lComLink2,2)*m_cm1*m_cm2));
	gsl_vector_set(out,3,(I1*u2 + pow(l1,2)*m_cm2*u2 + pow(lComLink1,2)*m_cm1*u2 + (pow(l1*lComLink2*m_cm2*qd_i2,2)*sin(2*q_i1 - 2*q_i2))/2 - l1*lComLink2*m_cm2*u1*cosq1mq2 + pow(l1,3)*lComLink2*pow(m_cm2*qd_i1,2)*sinq1mq2 - g*pow(l1,2)*lComLink2*pow(m_cm2,2)*sin(q_i2) - I1*g*lComLink2*m_cm2*sin(q_i2) + g*pow(l1,2)*lComLink2*pow(m_cm2,2)*cosq1mq2*sin(q_i1) - g*pow(lComLink1,2)*lComLink2*m_cm1*m_cm2*sin(q_i2) + I1*l1*lComLink2*m_cm2*pow(qd_i1,2)*sinq1mq2 + l1*pow(lComLink1,2)*lComLink2*m_cm1*m_cm2*pow(qd_i1,2)*sin(q_i1 - q_i2) + g*l1*lComLink1*lComLink2*m_cm1*m_cm2*cosq1mq2*sin(q_i1))/(I1*I2 + pow(l1*lComLink2*m_cm2,2) + I2*pow(l1,2)*m_cm2 + I2*pow(lComLink1,2)*m_cm1 + I1*pow(lComLink2,2)*m_cm2 - pow(l1*lComLink2*m_cm2,2)*pow(cosq1mq2,2) + pow(lComLink1*lComLink2,2)*m_cm1*m_cm2));

	// Set derivative of inputs to zero
	gsl_vector_set(out,4,0);
	gsl_vector_set(out,5,0);
}

void twolinknog(gsl_vector *in, gsl_vector *out)
{
	double q_i1, q_i2, qd_i1, qd_i2, u1, u2;

	double m_cm1 = 0.809;
	double m_cm2 = 0.852;

	double I1 = 0.1449;
	double I2 = 0.1635;

	double l1 = 0.3825;
	double lComLink1 = 0.0700;
	double lComLink2 = 0.2450;

	// Give trivial names to elements of input
	q_i1 = gsl_vector_get(in,0);
	q_i2 = gsl_vector_get(in,1);
	qd_i1 = gsl_vector_get(in,2);
	qd_i2 = gsl_vector_get(in,3);
	u1 = gsl_vector_get(in,4);
	u2 = gsl_vector_get(in,5);

	// Derivative of first two elements equals the third and fourth element
	gsl_vector_set(out,0,qd_i1);
	gsl_vector_set(out,1,qd_i2);

	// The double-dot equations
	gsl_vector_set(out,2,-(sin(2*q_i1 - 2*q_i2)*pow(l1*lComLink2*m_cm2*qd_i1,2) + 2*sin(q_i1 - q_i2)*l1*pow(lComLink2,3)*pow(m_cm2*qd_i2,2) + 2*I2*sin(q_i1 - q_i2)*l1*lComLink2*m_cm2*pow(qd_i2,2) + 2*u2*cos(q_i1 - q_i2)*l1*lComLink2*m_cm2 - 2*u1*pow(lComLink2,2)*m_cm2 - 2*I2*u1)/(2*I1*I2 + pow(l1*lComLink2*m_cm2,2) + 2*I2*pow(l1,2)*m_cm2 + 2*I2*pow(lComLink1,2)*m_cm1 + 2*I1*pow(lComLink2,2)*m_cm2 - pow(l1*lComLink2*m_cm2,2)*cos(2*q_i1 - 2*q_i2) + 2*pow(lComLink1*lComLink2,2)*m_cm1*m_cm2));
	gsl_vector_set(out,3,(sin(q_i1 - q_i2)*pow(l1,3)*lComLink2*pow(m_cm2*qd_i1,2) + (sin(2*q_i1 - 2*q_i2)*pow(l1*lComLink2*m_cm2*qd_i2,2))/2 + u2*pow(l1,2)*m_cm2 + m_cm1*sin(q_i1 - q_i2)*l1*pow(lComLink1,2)*lComLink2*m_cm2*pow(qd_i1,2) + I1*sin(q_i1 - q_i2)*l1*lComLink2*m_cm2*pow(qd_i1,2) - u1*cos(q_i1 - q_i2)*l1*lComLink2*m_cm2 + m_cm1*u2*pow(lComLink1,2) + I1*u2)/(I1*I2 + pow(l1*lComLink2*m_cm2,2) + I2*pow(l1,2)*m_cm2 + I2*pow(lComLink1,2)*m_cm1 + I1*pow(lComLink2,2)*m_cm2 - pow(l1*lComLink2*m_cm2*cos(q_i1 - q_i2),2) + pow(lComLink1*lComLink2,2)*m_cm1*m_cm2));

	// Set derivative of inputs to zero
	gsl_vector_set(out,4,0);
	gsl_vector_set(out,5,0);
}


