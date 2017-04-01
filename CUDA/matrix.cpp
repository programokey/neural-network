#include "matrix.h"
#include <cstdio>
#include <cassert>
#include <omp.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
const int threads_count = 12;

double mul_time = 0;
double max_time = 0;
double dot_mul_time = 0;
double sub_time = 0;
double add_equal_time = 0;
double div_equal_time = 0;

void print_time()
{
	printf("mul_time = %gs\n", mul_time);
	printf("max_time = %gs\n", max_time);
	printf("dot_mul_time = %gs\n", dot_mul_time);
	printf("sub_time = %gs\n", sub_time);
	printf("add_equal_time = %gs\n", add_equal_time);
	printf("div_equal_time = %gs\n", div_equal_time);
}
static std::mt19937 mt(time(NULL));
static std::normal_distribution<real> normal(0, 1);
matrix::matrix()
{
	this->m = this->n = 0;
	this->data = NULL;
}
matrix::matrix(int n)
{
	assert(n > 0);
	this->m = this->n = n;
	this->data = (real*)malloc(sizeof(real)*n*n);
}

matrix::matrix(int m, int n)
{
	assert(m > 0 && n > 0);
	this->m = m;
	this->n = n;
	this->data = (real*)malloc(sizeof(real)*m*n);
}

matrix::matrix(const matrix &a)
{
	this->m = a.m;
	this->n = a.n;
	this->data = (real*)malloc(sizeof(real)*a.m*a.n);
	memcpy(this->data, a.data, sizeof(real)*n*m);
}


real & matrix::operator()(int i)
{
	assert(i <= n*m && i > 0);
	return data[i - 1];
}

real & matrix::operator()(int i, int j)
{
	assert(i > 0 && i <= m && j > 0 && j <= n);
	i--;
	j--;
	return data[i*n + j];
}

matrix matrix::row(int k)
{
	assert(k > 0 && k <= m);
	matrix res(1,n);
	k = k*n - n;
	for (int i = 0; i < n; i++)
		res.data[i] = this->data[k + i];
	return res;
}

matrix matrix::row(int h, int k)
{
	assert(h > 0 && k >= h && k <= m);
	int len = k - h + 1;
	matrix res(len, n);
	len *= n;
	k = h*n - n;
	for (int i = 0; i < len; i++)
		res.data[i] = this->data[k + i];
	return res;
}

matrix matrix::column(int k)
{
	matrix res(m, 1);
	for (int i = 1; i <= m; i++)
		res(i, 1) = this->operator()(i, k);
	return res;
}

matrix matrix::column(int h, int k)
{
	matrix res(m, k - h + 1);
	for (int i = 1; i <= m; i++)
		for (int j = 1; j <= k - h + 1;j++)
		res(i, j) = this->operator()(i, j + h - 1);
	return res;
}

int matrix::row_num()
{
	return m;
}

int matrix::column_num()
{
	return n;
}

matrix matrix::transpose()
{
	matrix a(n,m);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++)
			a(i, j) = this->operator()(j, i);
	return a;
}

matrix& matrix::operator=(const matrix & a)
{
	if (&a != this) {
		free(this->data);
		this->data = (real*)malloc(sizeof(real)*a.m * a.n);
		this->m = a.m;
		this->n = a.n;
		memcpy(this->data, a.data, sizeof(real)*n*m);
	}
	return *this;
}

matrix operator+(matrix &a, matrix &b)
{
	if (b.n == 1) {
		matrix res(a.m, a.n);
		for (int i = 1; i <= a.m; i++)
			for (int j = 1; j <= a.n; j++)
				res(i, j) = a(i, j) + b(i);
		return res;
	}
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
			res.data[i] =a.data[i] + b.data[i];
	return res;
}

matrix operator+(matrix &a, real b)
{
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] + b;
	return res;
}

matrix operator-(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] - b.data[i];
	return res;
}

matrix operator-(matrix &a, real b)
{
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] - b;
	return res;
}

matrix operator*(matrix &a, matrix &b)
{
	if (b.n == 1 && a.m == b.m) {
		matrix res(a.m, a.n);
		for (int i = 1; i <= a.m; i++)
			for (int j = 1; j <= a.n; j++)
				res(i,j) = a(i, j) * b(i);
		return res;
	}
	double t1 = omp_get_wtime();
	assert(b.m == a.n);
	matrix res(a.m, b.n);
#pragma omp parallel for num_threads(threads_count < a.m? threads_count : a.m)
	for (int i = 1; i <= a.m; i++) {
#pragma omp parallel for num_threads(threads_count < b.n? threads_count : b.n)
		for (int j = 1; j <= b.n; j++) {
			res(i, j) = 0;
			for (int k = 1; k <= a.n; k++)
				res(i, j) += a(i, k)*b(k,j);
		}
	}
	double t2 = omp_get_wtime();
	mul_time += t2 - t1;
	return res;
}

matrix operator*(matrix &a, real b)
{
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] * b;
	return res;
}

matrix operator*(real a, matrix &b)
{
	matrix res(b.m, b.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < b.n*b.m; i++)
		res.data[i] = b.data[i] * a;
	return res;
}

matrix operator/(matrix &a, real b)
{
	matrix res(a.m, a.n);
	assert(b != 0);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i]/b;
	return res;
}

void matrix::operator+=(matrix &a)
{
	double t1 = omp_get_wtime();
	if (a.n == 1) {
#pragma omp parallel for num_threads(threads_count)
		for (int i = 1; i <= m; i++)
			for (int j = 1; j <= n; j++)
				this->operator()(i, j) += a(i);
		return;
	}
	assert(m == a.m && n == a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < n*m; i++)
		data[i] += a.data[i];
	double t2 = omp_get_wtime();
	add_equal_time += t2 - t1;
}

void matrix::operator+=(real a)
{
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < n*m; i++)
		data[i] += a;
}

void matrix::operator-=(matrix &a)
{
	double t1 = omp_get_wtime(),t2;
	if (a.n == 1) {
#pragma omp parallel for num_threads(threads_count)
		for (int i = 1; i <= m; i++)
			for (int j = 1; j <= n; j++)
				this->operator()(i, j) -= a(i);
		t2 = omp_get_wtime();
		sub_time += t2 - t1;
		return;
	}
	assert(a.m == m && a.n == n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < n*m; i++)
		data[i] -= a.data[i];
	t2 = omp_get_wtime();
	sub_time += t2 - t1;
}

void matrix::operator-=(real a)
{
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < n*m; i++)
		data[i] -= a;
}

void matrix::operator*=(matrix &a)
{
	assert(size(a, 2) == 1 && a.m == m);
	for (int i = 1; i < m; i++)
		for (int j = 1; j < n; j++)
			this->operator()(i, j) *= a(i);
}

void matrix::operator*=(real a)
{
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < n*m; i++)
		data[i] *= a;
}

void matrix::operator/=(matrix & a)
{
	if (a.n == 1) {
		double t1 = omp_get_wtime();
#pragma omp parallel for num_threads(threads_count)
		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				this->operator()(i, j) /= a(i);
			}
		}
		double t2 = omp_get_wtime();
		div_equal_time += t2 - t1;
		return;
	}
}

void matrix::operator/=(real a)
{
	assert(a != 0);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < n*m; i++)
		data[i] /= a;
}

void matrix::mask(real x)
{
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < n*m; i++)
		this->data[i] = this->data[i] > x ? this->data[i] : x;
}

void matrix::print()
{
	putchar('\n');
	for (int i = 1; i <= m; i++) {
		for (int j = 1; j <= n; j++) {
			printf("\t%8g", this->operator()(i, j));
		}
		putchar('\n');
	}
}

void matrix::foutput(FILE * f)
{
	fwrite(&m, sizeof(int), 1, f);
	fwrite(&n, sizeof(int), 1, f);
	fwrite(data, sizeof(real), m*n, f);
}

void matrix::fload(FILE * f)
{
	fread(&m, sizeof(int), 1, f);
	fread(&n, sizeof(int), 1, f);
	if (data != NULL)
		free(data);
	data = (real*)malloc(sizeof(real)*m*n);
	fread(data, sizeof(real), m*n, f);
}

real * matrix::get_data_ptr()
{
	return this->data;
}

matrix::~matrix()
{
	free(this->data);
	this->data = NULL;
	m = n = 0;
}

void mul(matrix & a, matrix & b, matrix & res)
{
	res.m = a.m;
	res.n = b.n;
	if (res.data != NULL)
		free(res.data);
	res.data = (real*)malloc(res. m*res.n*sizeof(real));
	for (int i = 1; i <= res.m; i++) {
		for (int j = 1; j <= res.n; j++) {
			res(i, j) = 0;
			for (int k = 1; k <= a.n; k++)
				res(i, j) += a(i, k) * b(k, j);
		}
	}
}

void mul(matrix & a, real & b, matrix & res)
{
	res.m = a.m;
	res.n = a.n;
	if (res.data != NULL)
		free(res.data);
	res.data = (real*)malloc(res.m*res.n*sizeof(real));
	for (int i = 0; i < res.m*res.n; i++)
		res.data[i] = a.data[i] * b;
}

matrix operator>(matrix &a, real b)
{
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] > b ? 1.0 : 0.0;
	return res;
}

matrix operator>(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] > b.data[i] ? 1.0 : 0.0;
	return res;
}

matrix operator>=(matrix &a, real b)
{
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] >= b ? 1.0:0.0;
	return res;
}

matrix operator>=(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] >= b.data[i] ? 1.0 : 0.0;
	return res;
}

matrix operator==(matrix &a, real b)
{
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] == b ? 1.0 : 0.0;
	return res;
}

matrix operator==(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] == b.data[i] ? 1.0 : 0.0;
	return res;
}

matrix max(real a, matrix &b)
{
	double t1 = omp_get_wtime();
	matrix res(b.m, b.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < b.n*b.m; i++)
		res.data[i] = a > b.data[i] ? a : b.data[i];
	double t2 = omp_get_wtime();
	max_time += t2 - t1;
	return res;
}

matrix max(matrix &a, real b)
{
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] > b ? a.data[i] : b;
	return res;
}

matrix max(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.n*a.m; i++)
		res.data[i] = a.data[i] > b.data[i] ? a.data[i] : b.data[i];
	return res;
}

int size(matrix &a, int d)
{
	assert(d == 1 || d == 2);
	if (d == 1)
		return a.m;
	else
		return a.n;
}

matrix size(matrix & a)
{
	matrix res(1,2);
	res(1) = a.m;
	res(2) = a.n;
	return res;
}

matrix sum(matrix &a)
{
	matrix res(a.m, 1);
	for (int i = 1; i <= a.m; i++) {
		res(i, 1) = 0;
		for (int j = 1; j <= a.n; j++) {
			res(i, 1) += a(i, j);
		}
	}
	return res;
}

matrix mean(matrix &a)
{
	matrix res(a.m, 1);
	for (int i = 1; i <= a.m; i++) {
		res(i, 1) = 0;
		for (int j = 1; j <= a.n; j++) {
			res(i, 1) += a(i, j);
		}
		res(i, 1) /= a.n;
	}
	return res;
}

matrix zeros(int n)
{
	matrix res(n);
	memset(res.data, 0, sizeof(real)*n*n);
	return res;
}

matrix zeros(matrix & a)
{
	int m = a(1);
	int n = a(2);
	matrix res(m, n);
	memset(res.data, 0, sizeof(real)*m*n);
	return res;
}

matrix zeros(int m, int n)
{
	matrix res(m, n);
	memset(res.data, 0, sizeof(real)*m*n);
	return res;
}

matrix ones(int n)
{
	matrix res(n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < n*n; i++)
		res.data[i] = 1.0;
	return res;
}

matrix ones(matrix & a)
{
	int m = a(1);
	int n = a(2);
	matrix res(m, n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < m*n; i++)
		res.data[i] = 1.0;
	return res;
}

matrix ones(int m, int n)
{
	matrix res(m, n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < m*n; i++)
		res.data[i] = 1.0;
	return res;
}

matrix sqrt(matrix &a)
{
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.m*a.n; i++)
		res.data[i] = sqrtl(a.data[i]);
	return res;
}

matrix dot_mul(matrix &a, matrix &b)
{
	double t1 = omp_get_wtime();
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.m*a.n; i++)
		res.data[i] = a.data[i] * b.data[i];
	double t2 = omp_get_wtime();
	dot_mul_time += t2 - t1;
	return res;
}

matrix dot_div(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
#pragma omp parallel for num_threads(threads_count)
	for (int i = 0; i < a.m*a.n; i++) {
		assert(b.data[i] != 0);
		res.data[i] = a.data[i] / b.data[i];
	}
	return res;
}

matrix randn(int m, int n)
{
	matrix res(m,n);
	for (int i = 0; i < m*n; i++)
		res.data[i] = normal(mt);
	return res;
}

matrix randn(int n)
{
	matrix res(n);
	for (int i = 0; i < n*n; i++)
		res.data[i] = normal(mt);
	return res;
}
