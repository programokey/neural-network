#ifndef SOLVER_H
#define SOLVER_H
#include "matrix.cuh"
class solver
{
public:
	virtual void signup(matrix* parameter, matrix* gradient);
	virtual void update();
	void set_learning_rate(real learning_rate)
	{
		this->learning_rate = learning_rate;
	}
protected:
	real learning_rate = 1e-3;
};
#endif // !SOLVER_H

