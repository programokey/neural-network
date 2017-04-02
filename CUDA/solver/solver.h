#ifndef SOLVER_H
#define SOLVER_H
#include "../matrix.cuh"
class solver
{
public:
	virtual void signup(matrix* parameter, matrix* gradient) {};
	virtual void update() {};
	virtual void regulation() {};
	void set_learning_rate(real learning_rate)
	{
		this->learning_rate = learning_rate;
	}
	void set_regularize_rate(real regularize_rate)
	{
		this->regularize_rate = regularize_rate;
	}
protected:
	real learning_rate = 1e-3;
	real regularize_rate = 1e-3;
};
#endif // !SOLVER_H

