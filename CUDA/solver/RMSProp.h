#ifndef RMSPROP_H
#define RMSPROP_H
#include "solver.h"
#include <vector>
class RMSProp :public solver
{
public:
	void signup(matrix* parameter, matrix* gradient);
	void update();
	void regulation();
private:
	std::vector<matrix*> theta;
	std::vector<matrix*> g;
	std::vector<matrix> cache;
	real epslion = 1e-8;
	int count = 0;
	real decay = 0.9;
};
#endif // !RMSPROP_H