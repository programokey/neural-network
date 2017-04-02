#include "test.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <omp.h>
#include <ctime>
#include "matrix.cuh"
#include "network.h"

void test_saved()
{

	std::mt19937_64 mt(time(NULL));
	std::uniform_real_distribution<real> uniform(-10, 10);
	int training_set_size = 10000;
	matrix training_set(3, training_set_size);
	matrix y(1, training_set_size);
	for (int p = 1; p <= training_set_size; p++)
		for (int k = 1; k <= 3; k++)
			training_set.set(k, p, uniform(mt));
	for (int p = 1; p <= training_set_size; p++) {
		real x = training_set(1, p);
		real u = training_set(2, p);
		real z = training_set(3, p);
		if (x*x + u*u + z*z < 60)
			y.set(p,1);
		else if (x*x + u*u + z*z < 120)
			y.set(p,2);
		else
			y.set(p,3);
	}
	network net("saved.net");
	matrix res = net.classify(training_set);
	int correct = sum(y == res)(1);
	puts("");
	printf("size is %d\n", training_set_size);
	printf("correct = %d\n", correct);
	printf("accuracy is %f%%\n", correct*100.0 / training_set_size);
	puts("");
}

real test(int training_set_size, int batch_size, real lambda, real learning_rate)
{
	std::mt19937_64 mt(time(NULL));
	std::uniform_real_distribution<real> uniform(-10, 10);
	matrix training_set(3, training_set_size);
	matrix y(1, training_set_size);
	for (int p = 1; p <= training_set_size; p++)
		for (int k = 1; k <= 3; k++)
			training_set.set(k, p, uniform(mt));
	for (int p = 1; p <= training_set_size; p++) {
		real x = training_set(1, p);
		real u = training_set(2, p);
		real z = training_set(3, p);
		if (x*x + u*u + z*z < 60)
			y.set(p, 1);
		else if (x*x + u*u + z*z < 120)
			y.set(p, 2);
		else
			y.set(p, 3);
	}
	std::vector<int> unit_count = { 3, 57, 43, 23, 13, 3 };
	network net(unit_count);
	net.set_train_set(training_set, y);
	net.set_batch_size(batch_size);
	net.set_regularize_rate(lambda);
	printf("begin training!\n");
	double t1 = omp_get_wtime();
	net.train();
	double t2 = omp_get_wtime();
	printf("train complete! time = %g s\n", t2 - t1);
	training_set_size = 100;
	training_set = zeros(3, training_set_size);
	y = zeros(1, training_set_size);
	for (int p = 1; p <= training_set_size; p++)
		for (int k = 1; k <= 3; k++)
			training_set.set(k, p, uniform(mt));
	for (int p = 1; p <= training_set_size; p++) {
		real x = training_set(1, p);
		real u = training_set(2, p);
		real z = training_set(3, p);
		if (x*x + u*u + z*z < 60)
			y.set(p, 1);
		else if (x*x + u*u + z*z < 120)
			y.set(p, 2);
		else
			y.set(p, 3);
	}
	training_set_size = 100;
	matrix res = net.classify(training_set);
	res = get_class(res);
	int correct = sum(y == res)(1);
	(y - res).print();
	puts("");
	printf("size is %d\n", training_set_size);
	printf("correct = %d\n", correct);
	real accuracy = correct*100.0 / training_set_size;
	printf("accuracy is %f%%\n", accuracy);
	puts("");
	net.save("saved.net");
	return accuracy;
}
