#include "RMSProp.h"

void RMSProp::signup(matrix * parameter, matrix * gradient)
{
	theta.push_back(parameter);
	g.push_back(gradient);
	cache.push_back(zeros(size(*parameter)));
	count++;
}

void RMSProp::update()
{
	for (int i = 0; i < count; i++) {
		cache[i] = decay*cache[i] + (1 - decay)*dot_mul(*g[i], *g[i]);
		*theta[i] -= learning_rate*(*g[i])/(sqrt(sum_all(cache[i])) + epslion);
	}
}

void RMSProp::regulation()
{
	for (int i = 0; i < count; i++)
		(*g[i]) += regularize_rate*(*theta[i]);
}

