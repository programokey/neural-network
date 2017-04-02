#include "AdaGrad.h"

void AdaGrad::signup(matrix * parameter, matrix * gradient)
{
	theta.push_back(parameter);
	g.push_back(gradient);
	count++;
}

void AdaGrad::update()
{
	matrix cache;
	for (int i = 0; i < count; i++) {
		//*theta[i] -= learning_rate*(*g[i])/(norm2(*g[i]) + epslion);
		*g[i] /= (sqrt(dot_mul(*g[i], *g[i])) + epslion);
		*theta[i] -= learning_rate*(*g[i]);
	}
}

void AdaGrad::regulation()
{
	for (int i = 0; i < count; i++)
		(*g[i]) += regularize_rate*(*theta[i]);
}