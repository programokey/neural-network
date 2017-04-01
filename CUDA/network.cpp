#include "network.h"
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <omp.h>
const int threads_count = 12;
network::network(std::vector<int> &unit_count)
{
	this->unit_count = unit_count;
	this->depth = unit_count.size();
	this->W.resize(depth - 1);
	for (int i = 0; i < depth - 1; i++)
		W[i] = randn(unit_count[i + 1], unit_count[i])*sqrt(2.0 / unit_count[i]);
	this->sigma.resize(depth - 1);
	this->mu.resize(depth - 1);
	this->gamma.resize(depth - 1);
	for (int i = 0; i < depth - 1; i++)
		this->gamma[i] = ones(unit_count[i], 1);
	this->beta.resize(depth - 1);
	for (int i = 0; i < depth - 1; i++)
		this->beta[i] = zeros(unit_count[i], 1);
}

void network::init_training()
{
	this->X.resize(depth);
	this->Y.resize(depth);
	this->m.resize(depth - 1);
	this->v.resize(depth - 1);
	this->dW.resize(depth - 1);
	this->dGamma.resize(depth - 1);
	this->dBeta.resize(depth - 1);
	this->mBeta.resize(depth - 1);
	this->mGamma.resize(depth - 1);
	this->vBeta.resize(depth - 1);
	this->vGamma.resize(depth - 1);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < depth - 1; i++)
		m[i] = v[i] = zeros(size(W[i]));
	for (int i = 1; i < depth - 1; i++) {
		mBeta[i] = vBeta[i] = zeros(size(beta[i]));
		mGamma[i] = vGamma[i] = zeros(size(gamma[i]));
	}
	char file[255];
	sprintf(file, "loss_%d.txt", time(NULL));
	lossf = fopen(file, "w");
}
network::network(const char * filename)
{
	FILE* f = fopen(filename, "rb");
	if (f == NULL)
		abort();
	fseek(f, 0, 0);
	fread(&depth, sizeof(int), 1, f);
	int count;
	unit_count.resize(depth);
	for (int i = 0; i < depth; i++) {
		fread(&count, sizeof(int), 1, f);
		unit_count[i] = count;
	}
	this->W.resize(depth - 1);
	this->sigma.resize(depth - 1);
	this->mu.resize(depth - 1);
	this->gamma.resize(depth - 1);
	this->beta.resize(depth - 1);
	for (int i = 0; i < depth - 1; i++) {
		W[i].fload(f);
		sigma[i].fload(f);
		mu[i].fload(f);
		gamma[i].fload(f);
		beta[i].fload(f);
	}
	fclose(f);
}
void network::train()
{
	double t1, t2;
	double forward_time, backward_time, regulation_time, adma_time;
	init_training();
	printf("batch size is %d\n",batch_size);
	printf("max it is %d\n", iteration);
	int64_t total = 0;
	BN(this->trainning_set, 0);
	for (int batch_num = 0; batch_num < (size(trainning_set, 2) / batch_size); batch_num++) {
		X[0] = trainning_set.column(batch_num*batch_size + 1, (batch_num + 1)*batch_size);
		y = label.column(batch_num*batch_size + 1, (batch_num + 1)*batch_size);
		Y[0] = X[0];
		int batch_it = 0;
		real loss = 1e23, prev_loss = 1e233;
		learning_rate = 3e-2*pow(0.8, batch_num);
		for (int t = 0; t < iteration; t++) {
			t1 = omp_get_wtime();
			forward();
			t2 = omp_get_wtime();
			forward_time = t2 - t1;
			prev_loss = loss;
			t1 = omp_get_wtime();
			loss = get_gradient();
			t2 = omp_get_wtime();
			backward_time = t2 - t1;
			if (loss > prev_loss + 1e-3)
				learning_rate *= decay;
			if (loss < 1e-6 || fabs(loss - prev_loss) < 1e-6)
				break;
			t1 = omp_get_wtime();
			regulation();
			t2 = omp_get_wtime();
			regulation_time = t2 - t1;
			t1 = omp_get_wtime();
			adma(t);
			t2 = omp_get_wtime();
			adma_time = t2 - t1;
			batch_it++;
			if (batch_it % 1000 == 0) {
				printf("forward_time = %g,\n backward_time = %g,\n regulation_time = %g,\nadma_time = %g\n",
					forward_time, backward_time, regulation_time, adma_time);
				printf("%d/%d\n", batch_it, iteration);
			}
		}
		total += batch_it;
		printf("batch = %d, iteration = %d\n", batch_num + 1,batch_it);
	}
	X[0] = trainning_set;
	forward();
	printf("total iteration= %I64d\n",total);
}

void network::set_train_set(const matrix &trainning_set, const matrix &label)
{
	this->trainning_set = trainning_set;
	this->label = label;
}

matrix network::classify(matrix & feature)
{
	matrix x = feature;
	x -= mu[0];
	x /= sigma[0];

	for (int i = 1; i < depth - 1; i++) {
		x = W[i - 1] * x;
		x -= mu[i];
		x /= sigma[i];
		x = x*gamma[i] + beta[i];
		x.mask(0);
	}
	x = W[depth - 2] * x;
	return x;
	//return get_class(x);
}

void network::save(const char * filename)
{
	FILE* f = fopen(filename, "wb");
	if(f == NULL)
		abort();
	fseek(f, 0, 0);
	fwrite(&depth, sizeof(int), 1, f);
	for(int i: unit_count)
		fwrite(&i, sizeof(int), 1, f);
	for (int i = 0; i < depth - 1;i++) {
		W[i].foutput(f);
		sigma[i].foutput(f);
		mu[i].foutput(f);
		gamma[i].foutput(f);
		beta[i].foutput(f);
	}
	fclose(f);
}

void network::load(const char * filename)
{
	FILE* f = fopen(filename, "rb");
	if (f == NULL)
		abort();
	fseek(f, 0, 0);
	fread(&depth, sizeof(int), 1, f);
	int count;
	unit_count.resize(depth);
	for (int i = 0; i < depth; i++) {
		fread(&count, sizeof(int), 1, f);
		unit_count[i] = count;
	}
	for (int i = 0; i < depth - 1; i++) {
		W[i].fload(f);
		sigma[i].fload(f);
		mu[i].fload(f);
		gamma[i].fload(f);
		beta[i].fload(f);
	}
	fclose(f);
}

void network::set_batch_size(int batch_size)
{
	this->batch_size = batch_size;
}

void network::set_lambda(real lambda)
{
	this->lambda = lambda;
}

void network::set_learning_rate(real learning_rate)
{
	this->learning_rate = learning_rate;
}

network::~network()
{
	this->dW.clear();
	this->W.clear();
	this->X.clear();
	this->m.clear();
	this->v.clear();
	this->dBeta.clear();
	this->dGamma.clear();
	this->beta.clear();
	this->gamma.clear();
	this->mBeta.clear();
	this->mGamma.clear();
	this->sigma.clear();
	this->mu.clear();
	fcloseall();
}

void network::adma(int t)
{
	real beta1t = beta1*pow(1 - 1e-8, t++);
//#pragma omp parallel for num_threads(threads_count)
	for (int i = 1; i < depth - 1; i++) {
		matrix mb, vb;
		mBeta[i] = beta1t*mBeta[i] + (1 - beta1t)*(dBeta[i]);
		mGamma[i] = beta1t*mGamma[i] + (1 - beta1t)*(dGamma[i]);
		vBeta[i] = beta2*vBeta[i] + (1 - beta2)*dot_mul(dBeta[i], dBeta[i]);
		vGamma[i] = beta2*vGamma[i] + (1 - beta2)*dot_mul(dGamma[i], dGamma[i]);

		vb = vBeta[i] / (1 - pow(beta2, t));
		mb = mBeta[i] / (1 - pow(beta1, t));
		mb /= (sqrt(vb) + 1e-7);
		mb *= learning_rate;
		beta[i] -= mb;

		vb = vGamma[i] / (1 - pow(beta2, t));
		mb = mGamma[i] / (1 - pow(beta1, t));
		mb /= (sqrt(vb) + 1e-7);
		mb *= learning_rate;
		gamma[i] -= mb;
	}
	for (int i = 0; i < depth - 1; i++) {
		matrix mb, vb;
		m[i] *= beta1t;
		m[i] += (1 - beta1t)*(dW[i]);
		v[i] *= beta2;
		v[i] += (1 - beta2)*dot_mul(dW[i], dW[i]);
		vb = v[i] / (1 - pow(beta2, t));
		mb = m[i] / (1 - pow(beta1, t));
		mb /= (sqrt(vb) + 1e-7);
		mb *= learning_rate;
		W[i] -= mb;
	}
}

void network::regulation()
{
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < depth - 1; i++)
		dW[i] += lambda*W[i];
	for (int i = 1; i < depth - 1; i++) {
		dBeta[i] += lambda*beta[i];
		dGamma[i] += lambda*gamma[i];
	}
}

real network::get_gradient()
{
	double t1, t2;
	real loss = 0.0;
	for (int i = 0; i < depth - 1; i++) {
		dBeta[i] = zeros(size(beta[i]));
		dGamma[i] = zeros(size(gamma[i]));
	}
	matrix dx;
	loss = get_dx(dx, X[depth - 1], y);
	//printf("%g\n",loss);
	for (int k = depth - 2; k >= 0; k--) {
		get_dW(dW[k], dx, Y[k]);
		update_dX1(W[k], dx, Y[k]);
		dBeta[k] += sum(dx);
		dGamma[k] += sum(dot_mul(dx, X[k]));
		update_dX2(dx,gamma[k], sigma[k]);
	}
	for (int i = 0; i < depth - 1; i++)
		dW[i] /= batch_size;
	for (int i = 1; i < depth - 1; i++) {
		dGamma[i] /= batch_size;
		dBeta[i] /= batch_size;
	}
	fprintf(lossf, "%g\n", loss);
	return loss;
}

matrix get_class(matrix & out)
{
	matrix index = ones(1, size(out,2));
	for (int j = 1; j <= size(out, 2); j++) {
		for (int i = 2; i <= size(out, 1); i++) {
			if (out(i, j) > out(index(j), j)) {
				index.set(j, i);
			}
				
		}
	}
	return index;
}

void network::forward()
{
	for (int i = 1; i < depth - 1; i++) {
		mul(W[i - 1], Y[i - 1], X[i]);
		BN(X[i], i);
		Y[i] = X[i]*gamma[i] + beta[i];
		Y[i].mask(0.0);
	}
	X[depth - 1] = W[depth - 2] * Y[depth - 2];
}

void network::BN(matrix &x, int layer)
{
	mu[layer] = mean(x);
	x -= mu[layer];
	sigma[layer] = sqrt(sum(dot_mul(x, x)) / size(x, 2));
	x /= sigma[layer];
}


