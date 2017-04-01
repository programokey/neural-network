#ifndef DISTRIBUTION_CLASSIFICATION
#define DISTRIBUTION_CLASSIFICATION
#include "matrix.cuh"
void distribution_classification(int training_set_size, int batch_size, real lambda, real learning_rate);
void test_distribution_classification(int test_set_size);
void retrain_distribution_classification(int training_set_size, int batch_size, real lambda, real learning_rate);
void test_distribution(int test_set_size);
#endif // !DISTRIBUTION_CLASSIFICATION

