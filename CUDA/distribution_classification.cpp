#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <omp.h>
#include <ctime>
#include "matrix.cuh"
#include "network.h"
#include "statistical.h"
#include <algorithm>
#include <cassert>

void distribution_classification(int training_set_size, int batch_size, real lambda, real learning_rate)
{
	std::default_random_engine engin(time(NULL));
	std::uniform_int_distribution<int> type(1,10);
	std::uniform_int_distribution<int> count(10, 100000);
	int feature_num = 14;
	std::vector<int> layers = { feature_num, 233, 131, 113, 71, 59, 31, 23, 10 };
	network dist_net(layers);
	matrix y(1, training_set_size);
	matrix training_set(feature_num, training_set_size);
	matrix feature(feature_num, 1);
	std::vector<real> statistical_data;
	for (int i = 1; i <= training_set_size; i++) {
		y.set(i, type(engin));
		int num = count(engin); 
		generate_statistical_data(y(i), num, statistical_data);
		get_feature(statistical_data, feature);
		for (int k = 1; k <= feature_num; k++)
			training_set.set(k, i, feature(k));
	}
	dist_net.set_train_set(training_set, y);
	dist_net.set_batch_size(batch_size);
	dist_net.set_regularize_rate(lambda);
	dist_net.set_learning_rate(learning_rate);
	dist_net.train();
	dist_net.save("training_result1.dn");

	//test the netwrok
	int test_set_size = 100;
	std::vector<real> test_data;
	matrix label(1, test_set_size);
	matrix test_set(feature_num, test_set_size);
	for (int i = 1; i <= test_set_size; i++) {
		label.set(i, type(engin));
		int num = count(engin);
		generate_statistical_data(label(i), num, test_data);
		get_feature(test_data, feature);
		for (int k = 1; k <= feature_num; k++)
			test_set.set(k, i, feature(k));
	}
	matrix res = dist_net.classify(test_set);
	int correct = sum(label == res)(1);
	puts("");
	printf("size is %d\n", test_set_size);
	printf("correct = %d\n", correct);
	printf("accuracy is %f%%\n", correct*100.0 / test_set_size);
	puts("");
}
void test_distribution_classification(int test_set_size)
{
	std::mt19937_64 engine(time(NULL));
	std::uniform_int_distribution<int> type(1, 10);
	std::uniform_int_distribution<int> count(10, 100000);
	int feature_num = 14;
	std::vector<std::vector<real>> test_data(test_set_size);
	matrix label(1, test_set_size);
	matrix test_set(feature_num, test_set_size);
	matrix feature(feature_num, 1);
	for (int i = 1; i <= test_set_size; i++) {
		label.set(i, type(engine));
		int num = count(engine);
		generate_statistical_data(label(i), num, test_data[i - 1]);
		get_feature(test_data[i - 1], feature);
		for (int k = 1; k <= feature_num; k++)
			test_set.set(k, i, feature(k));
	}
	network net("training_result.dn");
	matrix res = net.classify(test_set);
	int correct = sum(label == res)(1);
	puts("");
	printf("size is %d\n", test_set_size);
	printf("correct = %d\n", correct);
	printf("accuracy is %f%%\n", correct*100.0 / test_set_size);
	puts("");
}
void retrain_distribution_classification(int training_set_size, int batch_size, real lambda, real learning_rate)
{
	std::default_random_engine engin(time(NULL));
	std::uniform_int_distribution<int> type(1, 10);
	std::uniform_int_distribution<int> count(10, 100000);
	int feature_num = 14;
	network dist_net("training_result.dn");
	matrix y(1, training_set_size);
	matrix training_set(feature_num, training_set_size);
	matrix feature(feature_num, 1);
	std::vector<real> statistical_data;
	for (int i = 1; i <= training_set_size; i++) {
		y.set(i, type(engin));
		int num = count(engin);
		generate_statistical_data(y(i), num, statistical_data);
		get_feature(statistical_data, feature);
		for (int k = 1; k <= feature_num; k++)
			training_set.set(k, i, feature(k));
	}
	dist_net.set_train_set(training_set, y);
	dist_net.set_batch_size(batch_size);
	dist_net.set_regularize_rate(lambda);
	dist_net.set_learning_rate(learning_rate);
	dist_net.train();
	dist_net.save("training_result.dn");

	//test the netwrok
	int test_set_size = 100;
	std::vector<std::vector<real>> test_data(test_set_size);
	matrix label(1, test_set_size);
	matrix test_set(feature_num, test_set_size);
	for (int i = 1; i <= test_set_size; i++) {
		label.set(i, type(engin));
		int num = count(engin);
		generate_statistical_data(label(i), num, test_data[i - 1]);
		get_feature(test_data[i - 1], feature);
		for (int k = 1; k <= feature_num; k++)
			test_set.set(k, i, feature(k));
	}
	matrix res = dist_net.classify(test_set);
	int correct = sum(label == res)(1);
	puts("");
	printf("size is %d\n", test_set_size);
	printf("correct = %d\n", correct);
	printf("accuracy is %f%%\n", correct*100.0 / test_set_size);
	puts("");
}


matrix get_class1(matrix & out)
{
	matrix index = ones(1, size(out, 2));
#pragma omp parallel num_threads(threads_count)
	for (int j = 1; j <= size(out, 2); j++)
#pragma omp for 
		for (int i = 2; i <= size(out, 1); i++)
			if (out(i, j) > out(index(j), j))
				index.set(j, i);
	return index;
}

/*void test_distribution(int test_set_size)
{
	std::mt19937_64 engine(time(NULL));
	std::uniform_int_distribution<int> type(1, 10);
	std::uniform_int_distribution<int> count(10, 100000);
	int feature_num = 14;
	std::vector<real> test_data;
	matrix label(1, test_set_size);
	matrix test_set(feature_num, test_set_size);
	matrix feature(feature_num, 1);
	for (int i = 1; i <= test_set_size; i++) {
		label(i) = type(engine);
		int num = count(engine);
		generate_statistical_data(label(i), num, test_data);
		get_feature(test_data, feature);
		for (int k = 1; k <= feature_num; k++)
			test_set(k, i) = feature(k);
	}
	network net("training_result.dn");
	matrix res = net.classify(test_set);
	label.print();
	res.print();
	printf("%g", (sum(label == get_class1(res))(1)));
}*/
char* types[] = { "normal", "log normal", "uniform","poisson",
				  "exponential", "gamma", "binomial","chi_squared",
				  "stduent_t", "fisher_f" };
struct result {
	real value;
	char* type;
};
int cmp(result &a, result &b)
{
	return a.value > b.value;
}
real reload(real x)
{
	char s[32];
	sprintf(s, "%f", x);
	real res;
	sscanf(s, "%lf", &res);
	return res;
}
void test_distribution(int test_set_size)
{
	std::mt19937_64 engine(time(NULL));
	std::uniform_int_distribution<int> type(1, 10);
	std::uniform_int_distribution<int> count(10, 100000);
	int feature_num = 14;
	std::vector<real> test_data;
	matrix test_set(feature_num, test_set_size);
	matrix feature(feature_num, 1);
	network net("training_result.dn");
	int cnt = 0;
	for (int i = 0; i < test_set_size; i++) {
		real label = type(engine);
		int num = count(engine);
		generate_statistical_data(label, num, test_data);
		//int N = test_data.size();
		//for (int i = 0; i < N; i++)
			//test_data[i] = reload(test_data[i]);
		get_feature(test_data, feature);
		matrix res = net.classify(feature);
		result value[10];
		for (int i = 0; i < 10; i++) {
			value[i].type = types[i];
			value[i].value = res(i + 1);
		}
		std::sort(value, value + 10, cmp);
		res = get_class1(res);
		puts("");
		printf("label: %s\n", types[(int)label - 1]);
		char* type = types[(int)res(1)];
		assert(type == value[0].type);
		for (int i = 0; i < 4; i++) {
			printf("%8s\t%g\n",value[i].type, value[i].value);
		}
		if (res(1) == label) {
			cnt++;
			/*char file[255];
			sprintf(file, "sample/%s%d.txt", types[int(label) - 1], i);
			FILE* f = fopen(file, "w");
			for (real x : test_data)
				fprintf(f, "%f\n", x);
			fclose(f);*/
		}
			
		printf("correct %d/%d\n",cnt, i + 1);
		puts("");
		//system("pause");
	}
	printf("%g%%\n", cnt*100.0 / test_set_size);
}