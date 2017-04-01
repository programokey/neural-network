#ifndef ADAM_H
#define ADAM_H
#include "solver.h"
#include <vector>
class adam : public solver 
{
public:
	void signup(matrix* parameter, matrix* gradient);
	void update();
private:
	std::vector<matrix*> theta;
	std::vector<matrix*> g;
	std::vector<matrix>  m;
	std::vector<matrix>  v;
	real beta1 = 0.9;
	real beta2 = 0.999;
	real lambda = 1 - 1e-8;
	real epslion = 1e-8;
	int t = 0;
	int count = 0;
};
#endif // !ADAM_H

