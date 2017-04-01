#ifndef NETWORK_H
#define NETWORK_H
#include <vector>
#include "matrix.cuh"
class network
{
public:
	network(std::vector<int> &unit_count);
	network(const char* filename);
	void train();
	void set_train_set(const matrix &trainning_set, const matrix &label);
	matrix classify(matrix& feature);
	void save(const char* filename);
	void load(const char* filename);
	void set_batch_size(int batch_size);
	void set_lambda(real lambda);
	void set_learning_rate(real learning_rate);
	friend matrix get_class(matrix& out);
	~network();
private:
	void adma(int t);
	void regulation();
	real get_gradient();
	void forward();
	void BN(matrix& x, int layer);
	void init_training();
	int depth;
	std::vector<int> unit_count;
	int batch_size = 400;
	int iteration = 10000;
	real lambda = 1e-3;
	real learning_rate = 1e-3;
	real beta1 = 0.9;
	real beta2 = 0.9;//0.999;
	real decay = 0.9;
	matrix trainning_set;
	matrix label;
	matrix y;

	std::vector<matrix> X;
	std::vector<matrix> Y;
	std::vector<matrix> W;
	std::vector<matrix> dW;
	std::vector<matrix> m;
	std::vector<matrix> v;
	std::vector<matrix> sigma;
	std::vector<matrix> mu;
	std::vector<matrix> gamma;
	std::vector<matrix> dGamma;
	std::vector<matrix> mGamma;
	std::vector<matrix> vGamma;
	std::vector<matrix> beta;
	std::vector<matrix> dBeta;
	std::vector<matrix> mBeta;
	std::vector<matrix> vBeta;
	FILE* lossf = NULL;
	bool cal = false;
};
#endif // !NETWORK_H

/*
3      20,3  20     20,20 20    20,20  20     2,20  2
X0 -BN0-W0-> X1 -BN1-W1-> X2 -BN2-W2-> X3 -BN3-W3-> X4
[3,20,20,20,2]
*/