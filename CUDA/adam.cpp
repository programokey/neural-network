#include "adam.h"
#include <vector>
#include <cmath>

void adam::signup(matrix * parameter, matrix * gradient)
{
	theta.push_back(parameter);
	g.push_back(gradient);
	m.push_back(zeros(size(*parameter)));
	v.push_back(zeros(size(*parameter)));
	count++;
}

void adam::update()
{
	real beta1t = beta1*pow(lambda, t);
	t++;
	matrix mb, vb;
	for (int i = 0; i < count; i++) {
		m[i] = beta1t*m[i] + (1 - beta1t)*(*g[i]);
		v[i] = beta2*v[i] + (1 - beta2)*dot_mul(*g[i], *g[i]);
		mb = m[i] / (1 - pow(beta1, t));
		vb = v[i] / (1 - pow(beta2, t));
		mb /= (sqrt(vb) + epslion);
		mb *= learning_rate;
		*theta[i] = (*theta[i]) - mb;
	}
}
