#ifndef ADAGRAD_H
#define ADAGRAD_H
#include "solver.h"
#include <vector>
class AdaGrad : public solver
{
public:
	void signup(matrix* parameter, matrix* gradient);
	void update();
	void regulation();
private:
	std::vector<matrix*> theta;
	std::vector<matrix*> g;
	real epslion = 1e-8;
	int count = 0;
};
#endif // !ADAGRAD_H


