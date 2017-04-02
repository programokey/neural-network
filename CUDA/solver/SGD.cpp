#include "SGD.h"

void SGD::signup(matrix * parameter, matrix * gradient)
{
	theta.push_back(parameter);
	g.push_back(gradient);
	count++;
}

void SGD::update()
{
	matrix cache;
	for (int i = 0; i < count; i++) {
		*theta[i] -= learning_rate*(*g[i]);
	}
}

void SGD::regulation()
{
	for (int i = 0; i < count; i++)
		(*g[i]) += regularize_rate*(*theta[i]);
}
