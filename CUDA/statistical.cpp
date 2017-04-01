#include "matrix.cuh"
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <ctime>
#include "statistical.h"
void estimate_normal(std::vector<real> &statistical_data, real &mu, real &sigma)
{
	int n = statistical_data.size();
	mu = 0;
	sigma = 0;
	for (real x : statistical_data) {
		mu += x;
		sigma += x*x;
	}
	mu /= n;
	sigma -= mu*mu;
	sigma /= n - 1;
	sigma = sqrt(sigma);
}
void estimate_lognormal(std::vector<real> &statistical_data, real &mu, real &sigma)
{
	int n = statistical_data.size();
	mu = 0;
	sigma = 0;
	for (real x : statistical_data) {
		mu += log(x);
		sigma += log(x)*log(x);
	}
	mu /= n;
	sigma -= mu*mu;
	sigma /= n - 1;
	sigma = sqrt(sigma);
}
void estimate_poisson(std::vector<real> &statistical_data, real &lambda)
{
	int n = statistical_data.size();
	lambda = 0;
	for (real x : statistical_data)
		lambda += x;
	lambda /= n;
}
void estimate_exp(std::vector<real> &statistical_data, real &theta)
{
	int n = statistical_data.size();
	theta = 0;
	for (real x : statistical_data)
		theta += x;
	theta /= n;
}
void estimate_uniform(std::vector<real> &statistical_data, real &a, real& b)
{
	a = 1e233, b = -1e233;
	for (real x : statistical_data) {
		if (x < a)
			a = x;
		if (x > b)
			b = x;
	}
}
void estimate_gamma(std::vector<real> &statistical_data, real &alpha, real& beta)
{
	int n = statistical_data.size();
	real mu = 0;
	real sigma = 0;
	for (real x : statistical_data) {
		mu += x;
		sigma += x*x;
	}
	mu /= n;
	sigma -= mu*mu;
	sigma /= n - 1;
	sigma = sqrt(sigma);
	beta = mu / sigma;
	alpha = mu*beta;
}
real c(int n,int m)
{
	real res = 0;
	for (int i = 1; i < m; i++)
		res *= (n - i + 1) / i;
	return res;
}
void estimate_binomial(std::vector<real> &statistical_data, int &n, real& p)
{
	p = 0;
	std::set<int> value;
	real sum = 0;
	for (real x : statistical_data) {
		sum += x;
		value.insert(round(x));
	}
	int k = value.size();
	real min_v = 1e233;
	int min_n = k;
	for (n = min_n > 7 ? min_n - 5 : 2; n < min_n + 5; n++) {
		p = sum / statistical_data.size()*n;
		real v = log(p)*n*k + (k*n - sum)*log(1 - p);
		for (real m : statistical_data)
			v += log(c(n, round(m)));
		if (min_v < v) {
			min_n = n;
			min_v = v;
		}
	}
	n = min_n;
	p = sum / statistical_data.size()*n;
}
void estimate_chi_squared(std::vector<real> &statistical_data, int &n)
{
	n = statistical_data.size();
	real mu = 0;
	real sigma = 0;
	for (real x : statistical_data) {
		mu += x;
		sigma += x*x;
	}
	mu /= n;
	sigma -= mu*mu;
	sigma /= n - 1;
	sigma = sqrt(sigma);
	n = (sigma + mu) / 3;
}
void estimate_t(std::vector<real> &statistical_data, int &n)
{
	n = statistical_data.size();
	real mu = 0;
	real sigma = 0;
	for (real x : statistical_data) {
		mu += x;
		sigma += x*x;
	}
	mu /= n;
	sigma -= mu*mu;
	sigma /= n - 1;
	sigma = sqrt(sigma);
	if (fabs(sigma - 1) < 1e-2)
		n = 2;
	else if (sigma > 1)
		n = 2 * sigma / (sigma - 1);
	else
		n = 1;
}
void estimate_f(std::vector<real> &statistical_data, int &n1, int &n2)
{
	int n = statistical_data.size();
	std::map<real, int> freq;
	real mu = 0;
	real sigma = 0;
	for (real x : statistical_data) {
		mu += x;
		sigma += x*x;
		freq[x]++;
	}
	mu /= n;
	sigma -= mu*mu;
	sigma /= n - 1;
	sigma = sqrt(sigma);
	if (fabs(mu - 1) < 1e-2)
		n2 = 2;
	else if (mu > 1)
		n2 = 2 * mu / (mu - 1);
	else
		n2 = 1;
	int mode_freq = 0;
	real mode;
	for (std::pair<real, int> p : freq) {
		if (p.second > mode_freq) {
			mode = p.first;
			mode_freq = p.second;
		}
	}
	mode = mode*(n2 + 2) / n2;
	if (fabs(mode - 1) < 1e-2)
		n1 = 2;
	else if (mode < 1)
		n1 = 2 / (1 - mode);
	else
		n1 = 1;
}
matrix get_feature(std::vector<real> &statistical_data, matrix &feature)
{
	int n = statistical_data.size();
	real arithmetic_mean = 0, geometric_mean = 0, standard_deviation = 0, cv = 0, skewness = 0, kurtosis = 0;
	for (real x : statistical_data) {
		arithmetic_mean += x;
		geometric_mean += x > 0 ?log2(x):-100;
	}
	arithmetic_mean /= n;
	geometric_mean /= n;
	geometric_mean = pow(2, geometric_mean);
	for (real x : statistical_data) {
		real t = (x - arithmetic_mean);
		standard_deviation += t*t;
		skewness += t*t*t;
		kurtosis += t*t*t*t;
	}
	standard_deviation /= n - 1;
	skewness /= n*pow(standard_deviation, 1.5);
	kurtosis /= n*standard_deviation*standard_deviation;
	cv = arithmetic_mean / sqrt(standard_deviation);
	feature.set(1, arithmetic_mean);
	feature.set(2, geometric_mean);
	if (standard_deviation == 0) {
		printf("standard_deviation = 0");
		system("pause");
	}
		
	feature.set(3, standard_deviation);
	feature.set(4, cv);
	feature.set(5, skewness);
	feature.set(6, kurtosis);
	std::sort(statistical_data.begin(), statistical_data.end());
	for (int i = 0; i < 8; i++)
		feature.set(7 + i, statistical_data[n*i / 8]);
	return feature;
}
void generate_statistical_data(int type,int count, std::vector<real> &data)
{
	std::default_random_engine engine(time(NULL));
	data.resize(count);
	switch (type)
	{
	case 1:
	{
		std::uniform_real_distribution<real> parameter1(-20, 20);
		std::uniform_real_distribution<real> parameter2(1e-3, 10000);
		std::normal_distribution<real> normal(parameter1(engine), parameter2(engine));
		for (int i = 0; i < count; i++)
			data[i] = normal(engine);
		break;
	}
	case 2:
	{
		std::uniform_real_distribution<real> parameter1(-3, 10);
		std::uniform_real_distribution<real> parameter2(1, 6);
		std::lognormal_distribution<> lognormal(parameter1(engine), parameter2(engine));
		for (int i = 0; i < count; i++)
			data[i] = lognormal(engine);
		break;
	}
	case 3:
	{
		std::uniform_real_distribution<> parameter(-100, 100);
		double min = parameter(engine);
		double max;
		do
			max = parameter(engine);
		while (min >= max);
			
		std::uniform_real_distribution<> uniform(min, max);
		for (int i = 0; i < count; i++)
			data[i] = uniform(engine);
		break;
	}
	case 4:
	{
		std::uniform_real_distribution<real> parameter(0, 10000);
		std::poisson_distribution<int> poisson(parameter(engine));
		for (int i = 0; i < count; i++)
			data[i] = poisson(engine);
		break;
	}
	case 5:
	{
		std::uniform_real_distribution<> parameter(1e-6, 20);
		std::exponential_distribution<> exp(parameter(engine));
		for (int i = 0; i < count; i++)
			data[i] = exp(engine);
		break;
	}
	case 6:
	{
		std::uniform_real_distribution<> parameter(1e-6, 10);
		std::gamma_distribution<> gamma(parameter(engine), parameter(engine));
		for (int i = 0; i < count; i++)
			data[i] = gamma(engine);
		break;
	}
	case 7:
	{
		std::uniform_int_distribution<int> parameter1(1, 1000);
		std::uniform_real_distribution<real> parameter2(1e-6, 1);
		int t = parameter1(engine);
		real p = parameter2(engine);
		std::binomial_distribution<int>binomial(t, p);
		for (int i = 0; i < count; i++)
			data[i] = binomial(engine);
		break;
	}
	case 8:
	{
		std::uniform_int_distribution<int> parameter(1, 1000);
		std::chi_squared_distribution<real> chi_squared(parameter(engine));
		for (int i = 0; i < count; i++)
			data[i] = chi_squared(engine);
		break;
	}
	case 9:
	{
		std::uniform_int_distribution<int> parameter(1, 1000);
		std::student_t_distribution<> t(parameter(engine));
		for (int i = 0; i < count; i++)
			data[i] = t(engine);
		break;
	}
	case 10:
	{
		std::uniform_int_distribution<int> parameter(1, 1000);
		std::fisher_f_distribution<> f(parameter(engine), parameter(engine));
		for (int i = 0; i < count; i++)
			data[i] = f(engine);
		break;
	}
	default:
		break;
	}
}
/*
normal:
	mean = [-20,20]
	variance = [1e-3, 1000]
log_normal
	mean = [-3,10]
	variance = [1, 6]
uniform
	min = [-100, 100];
	max = [-100, 100];
poisson
	lambda = [0, 10000]
exponential
	lambda = [1e-6, 20]
gamma
	alpha = [1e-6, 10]
	beta = [1e-6, 10]
binomial
	t = [1, 1000]
	p = [1e-6, 1]
chi_squared
	n = [1, 1000]
student_t
	n = [1, 1000]
fisher_f
	n1 = [1, 1000]
	n2 = [1, 1000]
*/