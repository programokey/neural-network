#include <omp.h>
#include "matrix.cuh"
#include <cstdio>
#include <cassert>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include "matrix.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
const int block_size = 256;

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
	cudaMalloc(&(this->data), sizeof(real)*n*n);
}

matrix::matrix(int m, int n)
{
	assert(m > 0 && n > 0);
	this->m = m;
	this->n = n;
	cudaMalloc(&(this->data), sizeof(real)*m*n);
}

matrix::matrix(const matrix &a)
{
	this->m = a.m;
	this->n = a.n;
	cudaMalloc(&(this->data), sizeof(real)*a.m*a.n);
	cudaMemcpy(this->data, a.data, sizeof(real)*n*m, cudaMemcpyDeviceToDevice);
}

int matrix::row_num()
{
	return m;
}

int matrix::column_num()
{
	return n;
}
__global__ void transpose_kernel(real* x, real* y, int m, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < m*n; i += stride) {
		int k = i / n;
		int h = i%n;
		y[h*m + k] = x[k*n + h];
	}
}
matrix matrix::transpose()
{
	matrix res(n, m);
	int numBlocks = (n*m + block_size - 1) / block_size;
	transpose_kernel <<<numBlocks, block_size >>>(this->data, res.data, m, n);
	return res;
}


matrix matrix::row(int k)
{
	assert(k > 0 && k <= m);
	matrix res(1, n);
	cudaMemcpy(res.data, data + n*(k - 1), sizeof(real)*n, cudaMemcpyDeviceToDevice);
	return res;
}

matrix matrix::row(int h, int k)
{
	assert(h > 0 && k >= h && k <= m);
	int len = k - h + 1;
	matrix res(len, n);
	real* tmp = new real[len*n];
	cudaMemcpy(res.data, data + n*(h - 1), sizeof(real)*n*len, cudaMemcpyDeviceToDevice);
	return res;
}
__global__ void get_column(real* x,real *y, int m, int n, int k, int h)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int len = (h - k + 1);
	for (int g = index; g < m*len; g+=stride) {
		int i = g / len;
		int j = g%len;
		y[g] = x[i*n + j + k];
	}
}
matrix matrix::column(int k)
{
	matrix res(m, 1);
	k--;
	int numBlocks = (m + block_size - 1) / block_size;
	get_column<<<numBlocks, block_size>>>(data, res.data, m, n, k, k);;
	return res;
}

matrix matrix::column(int h, int k)
{
	assert(k >= h);
	k--, h--;
	matrix res(m, k - h + 1);
	int numBlocks = (m*(k - h + 1) + block_size - 1) / block_size;
	get_column << <numBlocks, block_size >> >(data, res.data, m, n, h, k);
	return res;
}
real matrix::get(int k)
{
	real y;
	cudaMemcpy(&y, data + k - 1, sizeof(real), cudaMemcpyDeviceToHost);
	return y;
}

real matrix::get(int k, int h)
{
	k--;
	h--;
	real y;
	cudaMemcpy(&y, data + k*n + h, sizeof(real), cudaMemcpyDeviceToHost);
	return y;
}

void matrix::set(int k, real y)
{
	cudaMemcpy(data + k - 1, &y, sizeof(real), cudaMemcpyHostToDevice);
}

void matrix::set(int k, int h, real y)
{
	k--;
	h--;
	cudaMemcpy(data + k*n + h, &y, sizeof(real), cudaMemcpyHostToDevice);
}

real matrix::operator()(int k)
{
	return get(k);
}

real matrix::operator()(int k, int h)
{
	return get(k, h);
}


matrix& matrix::operator=(const matrix & a)
{
	if (&a != this) {
		if(this->data != NULL)
			cudaFree(this->data);
		//this->data = a.data;
		cudaMalloc(&(this->data), sizeof(real)*a.m * a.n);
		this->m = a.m;
		this->n = a.n;
		cudaMemcpy(this->data, a.data, sizeof(real)*n*m, cudaMemcpyDeviceToDevice);
	}
	return *this;
}
__global__ void add(real* x, real* y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] + y[i];
}
__global__ void add_column(real *x, real* y, int m, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < m*n; k+=stride) {
		int i = k / n;
		x[k] += y[i];
	}
}
matrix operator+(matrix &a, matrix &b)
{
	if (b.n == 1 && a.m == b.m) {
		matrix res = a;
		int numBlocks = (a.m*a.n + block_size - 1) / block_size;
		add_column << <numBlocks, block_size >> >(res.data, b.data, a.m, a.n);
		return res;
	}
	else {
		assert(a.m == b.m && a.n == b.n);
		matrix res(a.m, a.n);
		int numBlocks = (a.n*a.m + block_size - 1) / block_size;
		add << <numBlocks, block_size >> >(a.data, b.data, res.data, a.m * a.n);
		return res;
	}
	
}
__global__ void add(real* x, real y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] + y;
}
matrix operator+(matrix &a, real b)
{
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	add << <numBlocks, block_size>> >(a.data, b, res.data, a.m*a.n);
	return res;
}
__global__ void sub(real* x, real* y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] - y[i];
}
matrix operator-(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	sub << <numBlocks, block_size >> >(a.data, b.data, res.data, a.m * a.n);
	return res;
}
__global__ void sub(real* x, real y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] - y;
}
matrix operator-(matrix &a, real b)
{
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	sub << <numBlocks, block_size >> >(a.data, b, res.data, a.m*a.n);
	return res;
}
__global__ void mul_column(real *x, real* y, int m, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < m*n; k+=stride) {
		int i = k / n;
		x[k] *= y[i];
	}
}
__global__ void product(real* x, real *y, real* z,int m, int p, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int h = index; h < m*n; h += stride) {
		int i = h / n;
		int j = h%n;
		z[h] = 0;
		for (int k = 0; k < p; k++)
			z[h] += x[i*p + k] * y[k*n + j];
	}
}
matrix operator*(matrix & a, matrix & b)
{
	if (b.n == 1 && a.m == b.m) {
		matrix res = a;
		int numBlocks = (a.m*a.n + block_size - 1) / block_size;
		mul_column<<<numBlocks, block_size>>>(res.data, b.data, a.m, a.n);
		return res;
	}
	else {
		assert(a.n == b.m);
		matrix res(a.m, b.n);
		int numBlocks = (a.m*b.n + block_size - 1) / block_size;
		product << <numBlocks, block_size >> >(a.data, b.data, res.data, a.m, a.n, b.n);
		return res;
	}
}
__global__ void transposed_product_second(real* x, real* y, real* z, int m, int p, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int h = index; h < m*n; h += stride) {
		int i = h / n;
		int j = h%n;
		z[h] = 0;
		for (int k = 0; k < p; k++)
			z[h] += x[i*p + k] * y[j*p + k];
	}
}
matrix transposed_mul(matrix & a, matrix & b)
{
	assert(a.n == b.n);
	matrix res(a.m, b.m);
	int numBlocks = (a.m*b.m + block_size - 1) / block_size;
	transposed_product_second<<<numBlocks, block_size>>>(a.data, b.data, res.data, a.m, a.n, b.m);
	return res;
}
__global__ void transposed_product_first(real* x, real* y, real* z, int m, int p, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int h = index; h < m*n; h += stride) {
		int i = h / n;
		int j = h%n;
		z[h] = 0;
		for (int k = 0; k < p; k++)
			z[h] += x[k*m + i] * y[k*n + j];
	}
}
void transposed_mul(matrix & a, matrix & b, matrix & res)
{
	assert(a.m == b.m);
	if (res.data != NULL)
		cudaFree(res.data);
	res.m = a.n;
	res.n = b.n;
	cudaMalloc(&res.data, sizeof(real)*a.n*b.n);
	int numBlocks = (a.n*b.n + block_size - 1) / block_size;
	transposed_product_first<< <numBlocks, block_size >> >(a.data, b.data, res.data, a.n, a.m, b.n);
}
__global__ void multiply(real* x, real y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] * y;
}
matrix operator*(matrix &a, real b)
{
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	multiply << <numBlocks, block_size>> >(a.data, b, res.data, a.m*a.n);
	return res;
}

matrix operator*(real a, matrix &b)
{
	matrix res(b.m, b.n);
	int numBlocks = (b.n*b.m + block_size - 1) / block_size;
	multiply << <numBlocks, block_size >> >(b.data, a, res.data, b.m*b.n);
	return res;
}
__global__ void divide(real* x, real* y, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] /= y[i];
}
__global__ void divide(real* x, real y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] / y;
}
__global__ void divide(real* x, real y, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] /= y;
}
void matrix::operator/=(real a)
{
	assert(a != 0);
	int numBlocks = (n*m + block_size - 1) / block_size;
	divide << <numBlocks, block_size >> >(data, a, m*n);
}
matrix operator/(matrix &a, real b)
{
	matrix res(a.m, a.n);
	assert(b != 0);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	divide << <numBlocks, block_size >> >(a.data, b, res.data, a.m*a.n);
	return res;
}
__global__ void add(real* x, real* y, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] += y[i];
}

void matrix::operator+=(matrix &a)
{
	if (a.n == 1) {
		assert(m == a.m);
		int numBlocks = (a.m*a.n + block_size - 1) / block_size;
		add_column << <numBlocks, block_size >> >(this->data, a.data, m, n);
	}
	else {
		assert(m == a.m && n == a.n);
		int numBlocks = (a.n*a.m + block_size - 1) / block_size;
		add << <numBlocks, block_size >> >(this->data, a.data, m*n);
	}
}
__global__ void add(real* x, real y, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] += y;
}
void matrix::operator+=(real a)
{
	int numBlocks = (n*m + block_size - 1) / block_size;
	add << <numBlocks, block_size >> >(this->data, a, m*n);
}
__global__ void sub(real* x, real* y, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] -= y[i];
}
__global__ void sub_column(real *x, real* y, int m, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < m*n; k+=stride) {
		int i = k / n;
		x[k] -= y[i];
	}
}
void matrix::operator-=(matrix &a)
{
	if (a.n == 1) {
		assert(a.m == m);
		int numBlocks = (m*n + block_size - 1) / block_size;
		sub_column << <numBlocks, block_size >> >(this->data, a.data, m, n);
	}
	else {
		assert(a.m == m && a.n == n);
		int numBlocks = (a.n*a.m + block_size - 1) / block_size;
		sub << <numBlocks, block_size >> >(data, a.data, m*n);
	}
}

void matrix::operator-=(real a)
{
	int numBlocks = (n*m + block_size - 1) / block_size;
	add << <numBlocks, block_size >> >(this->data, -a, m*n);
}

void matrix::operator*=(matrix &a)
{
	assert(a.n == 1 && a.m == m);
	int numBlocks = (n*m + block_size - 1) / block_size;
	mul_column << <numBlocks, block_size >> >(this->data, a.data, m, n);
}
__global__ void multiply(real* x, real y, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] *= y;
}
void matrix::operator*=(real a)
{
	int numBlocks = (n*m + block_size - 1) / block_size;
	multiply << <numBlocks, block_size >> >(this->data, a, m*n);
}

__global__ void divide_column(real *x,real* y, int m, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < m*n; k+=stride) {
		int i = k / n;
		x[k] /= y[i];
	}
}
void matrix::operator/=(matrix & a)
{
	if (a.n == 1) {
		assert(a.m == m);
		int numBlocks = (m*n + block_size - 1) / block_size;
		divide_column << <numBlocks, block_size >> >(this->data, a.data, m, n);
	}else if (a.m == m && a.n == n) {
		int numBlocks = (n*m + block_size - 1) / block_size;
		divide << <numBlocks, block_size >> >(this->data, a.data, m*n);
	}	
}

__global__ void cut_off(real *x, real y, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] = x[i] >= y ? x[i] : 0.0f;
}
void matrix::mask(real x)
{
	int numBlocks = (n*m + block_size - 1) / block_size;
	cut_off << <numBlocks, block_size >> >(data, x, m*n);
}

void matrix::print()
{
	putchar('\n');
	real* tmp = (real*)malloc(sizeof(real)*m*n);
	cudaMemcpy(tmp, data, sizeof(real)*m*n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("\t%8g", tmp[i*n + j]);
		}
		putchar('\n');
	}
	free(tmp);
}

void matrix::foutput(FILE * f)
{
	fwrite(&m, sizeof(int), 1, f);
	fwrite(&n, sizeof(int), 1, f);
	real* tmp = (real*)malloc(sizeof(real)*m*n);
	cudaMemcpy(tmp, data, sizeof(real)*m*n, cudaMemcpyDeviceToHost);
	fwrite(tmp, sizeof(real), m*n, f);
	free(tmp);
}

void matrix::fload(FILE * f)
{
	fread(&m, sizeof(int), 1, f);
	fread(&n, sizeof(int), 1, f);
	real* tmp = (real*)malloc(sizeof(real)*m*n);
	fread(tmp, sizeof(real), m*n, f);
	if (data != NULL)
		cudaFree(data);
	cudaMalloc(&data, sizeof(real)*m*n);
	cudaMemcpy(data, tmp, sizeof(real)*m*n, cudaMemcpyHostToDevice);
	free(tmp);
}

matrix::~matrix()
{
	if(this->data != NULL)
		cudaFree(this->data);
	this->data = NULL;
	m = n = 0;
}

void mul(matrix & a, matrix & b, matrix & res)
{
	assert(a.n == b.m);
	if (res.data != NULL)
		cudaFree(res.data);
	res.m = a.m;
	res.n = b.n;
	cudaMalloc(&res.data, sizeof(real)*a.m*b.n);
	int numBlocks = (a.m*b.n + block_size - 1) / block_size;
	product << <numBlocks, block_size >> >(a.data, b.data, res.data, a.m, a.n, b.n);
}

void mul(matrix & a, real & b, matrix & res)
{
	res.m = a.m;
	res.n = a.n;
	if (res.data != NULL)
		free(res.data);
	cudaMalloc(&res.data, res.m*res.n*sizeof(real));
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	multiply << <numBlocks, 256 >> >(a.data, b, res.data, a.m*a.n);
}

__global__ void greater_than(real* x, real y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] > y ? 1.0f : 0.0f;
}
matrix operator>(matrix &a, real b)
{
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	greater_than << <numBlocks,block_size >> >(a.data, b, res.data, a.m*a.n);
	return res;
}
__global__ void greater_than(real* x, real* y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] > y[i] ? 1.0f : 0.0f;
}
matrix operator>(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	greater_than << <numBlocks, block_size >> >(a.data, b.data, res.data, a.m*a.n);
	return res;
}

__global__ void greater_equal(real* x, real y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] >= y ? 1.0f : 0.0f;
}
matrix operator>=(matrix &a, real b)
{
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	greater_equal << <numBlocks, block_size >> >(a.data, b, res.data, a.m*a.n);
	return res;
}
__global__ void greater_equal(real* x, real* y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] >= y[i] ? 1.0f : 0.0f;
}
matrix operator>=(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	greater_equal << <numBlocks, block_size >> >(a.data, b.data, res.data, a.m*a.n);
	return res;
}
__global__ void equal(real* x, real y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] == y ? 1.0f : 0.0f;
}
matrix operator==(matrix &a, real b)
{
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	equal << <numBlocks, block_size >> >(a.data, b, res.data, a.m*a.n);
	return res;
}
__global__ void equal(real* x, real* y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] == y[i] ? 1.0f : 0.0f;
}
matrix operator==(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	equal << <numBlocks, block_size>> >(a.data, b.data, res.data, a.m*a.n);
	return res;
}
__global__ void maximum(real* x, real y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] > y ? x[i] : y;
}
__global__ void maximum(real* x, real* y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] > y[i]?x[i] : y[i];
}
matrix max(real a, matrix &b)
{
	matrix res(b.m, b.n);
	int numBlocks = (b.n*b.m + block_size - 1) / block_size;
	maximum << <numBlocks, block_size>> >(b.data, a, res.data, b.m*b.n);
	return res;
}

matrix max(matrix &a, real b)
{
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	maximum << <numBlocks, block_size>> >(a.data, b, res.data, a.m*a.n);
	return res;
}

matrix max(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	maximum << <numBlocks, block_size >> >(a.data, b.data, res.data, a.m*a.n);
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
	matrix res(1, 2);
	real tmp[2] = {(real)a.m, (real)a.n};
	cudaMemcpy(res.data, tmp, 2 * sizeof(real), cudaMemcpyHostToDevice);
	return res;
}
__global__ void sum_kernel(real* x, real* y, int m, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < m; i += stride) {
		y[i] = 0;
		for (int j = 0; j < n; j++) {
			y[i] += x[i*n + j];
		}
	}
}
matrix sum(matrix & a)
{
	matrix res(a.m, 1);
	int numBlocks = (a.m + block_size - 1) / block_size;
	sum_kernel<<<numBlocks, block_size>>>(a.data, res.data, a.m, a.n);
	return res;
}
__global__ void mean_kernel(real* x, real* y, int m, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < m; i += stride) {
		y[i] = 0;
		for (int j = 0; j < n; j++) {
			y[i] += x[i*n + j];
		}
		y[i] /= n;
	}
}
matrix mean(matrix & a)
{
	matrix res(a.m, 1);
	int numBlocks = (a.m + block_size - 1) / block_size;
	mean_kernel << <numBlocks, block_size >> >(a.data, res.data, a.m, a.n);
	return res;
}

__global__ void set(real* x, real y, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] = y;
}
matrix zeros(int n)
{
	matrix res(n);
	int numBlocks = (n*n + block_size - 1) / block_size;
	set << <numBlocks, block_size >> >(res.data, 0.0f, n*n);
	return res;
}

matrix zeros(matrix & a)
{
	real tmp[2];
	cudaMemcpy(tmp, a.data, 2*sizeof(real), cudaMemcpyDeviceToHost);
	int m = tmp[0];
	int n = tmp[1];
	matrix res(m, n);
	int numBlocks = (m*n + block_size - 1) / block_size;
	set << <numBlocks, block_size >> >(res.data, 0.0f, m*n);
	return res;
}

matrix zeros(int m, int n)
{
	matrix res(m, n);
	int numBlocks = (m*n + block_size - 1) / block_size;
	set << <numBlocks, block_size >> >(res.data, 0.0f, m*n);
	return res;
}
matrix ones(int n)
{
	matrix res(n);
	int numBlocks = (n*n + block_size - 1) / block_size;
	set << <numBlocks, block_size >> >(res.data, 1.0f, n*n);
	return res;
}
matrix ones(int m, int n)
{
	matrix res(m, n);
	int numBlocks = (m*n + block_size - 1) / block_size;
	set << <numBlocks, block_size >> >(res.data, 1.0f, m*n);
	return res;
}
matrix ones(matrix & a)
{
	real tmp[2];
	cudaMemcpy(tmp, a.data, 2*sizeof(real), cudaMemcpyDeviceToHost);
	int m = tmp[0];
	int n = tmp[1];
	matrix res(m, n);
	int numBlocks = (m*n + block_size - 1) / block_size;
	set << <numBlocks, block_size >> >(res.data, 1.0f, m*n);
	return res;
}
__global__ void square_root(real* x, real* y, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = gridDim.x*blockDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = sqrt((double)x[i]);
}
matrix sqrt(matrix &a)
{
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	square_root << <numBlocks, block_size>> >(a.data, res.data, a.m*a.n);
	return res;
}
__global__ void multiply(real* x, real* y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] * y[i];
}
matrix dot_mul(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	multiply << <numBlocks, block_size >> >(a.data, b.data, res.data, a.m*a.n);
	return res;
}
__global__ void divide(real* x, real* y, real* z, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] / y[i];
}
matrix dot_div(matrix &a, matrix &b)
{
	assert(a.m == b.m && a.n == b.n);
	matrix res(a.m, a.n);
	int numBlocks = (a.n*a.m + block_size - 1) / block_size;
	divide << <numBlocks, block_size >> >(a.data, b.data, res.data, a.m*a.n);
	return res;
}
__global__ void get_dx_kernel(real* dx, real* X, real* y, real* loss, int m, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int p = index; p < n; p+= stride) {
		loss[p] = 0;
		for (int i = 0; i < m; i++) {
			int k = y[p] - 1;
			if (i != k && X[i*n + p] + 1 > X[k*n + p]) {
				dx[i*n + p]++;
				dx[k*n + p]--;
				loss[p] += X[i*n + p] + 1 - X[k*n + p];
			}
		}
	}
}
real get_dx(matrix& dx, matrix& X, matrix & y)
{
	if (dx.data != NULL)
		cudaFree(dx.data);
	dx.m = X.m;
	dx.n = X.n;
	matrix loss(1, dx.n);
	cudaMalloc(&dx.data, sizeof(real)*dx.m*dx.n);
	int numBlocks = (dx.n + block_size - 1) / block_size;
	set << <numBlocks, block_size >> >(dx.data, 0, dx.m*dx.n);
	get_dx_kernel<<<numBlocks, block_size>>>(dx.data, X.data, y.data, loss.data, dx.m, dx.n);
	return sum(loss)(1);
}
__global__ void get_dW_kernel(real* dW, real* dx, real* Y, int m, int n, int b)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < m*n; k += stride) {
		int i = k / n;
		int j = k%n;
		dW[k] = 0;
		for (int h = 0; h < b; h++)
			dW[k] += dx[i*b + h] * Y[j*b + h];
	}
}
void get_dW(matrix & dW, matrix & dx, matrix & Y)
{
	dW.m = dx.m;
	dW.n = Y.m;
	if (dW.data != NULL)
		cudaFree(dW.data);
	cudaMalloc(&dW.data, sizeof(real)*dW.m*dW.n);
	int numBlocks = (dW.n*dW.m + block_size - 1) / block_size;
	get_dW_kernel<<<numBlocks,block_size>>>(dW.data, dx.data, Y.data, dW.m, dW.n, Y.n);
}
__global__ void update_dX1_kernel(real* dx,real* dx_next, real* W, real* Y, int m, int n , int b)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < n*b; k += stride) {
		int i = k / b;
		int j = k%b;
		dx_next[k] = 0;
		if (Y[k] <= 0)
			break;
		for (int h = 0; h < m; h++)
			dx_next[k] += W[h*n + i] * dx[h*b + j];
	}
}
void update_dX1(matrix & W, matrix & dx, matrix & Y)
{
	real* dx_next;
	cudaMalloc(&dx_next, sizeof(real)*W.n*dx.n);
	assert(dx.m == W.m && dx.n == Y.n && Y.m == W.n);
	int numBlocks = (W.n*dx.n + block_size - 1) / block_size;
	update_dX1_kernel<<<numBlocks,block_size>>>(dx.data, dx_next, W.data, Y.data, W.m, W.n, dx.n);
	cudaFree(dx.data);
	dx.data = dx_next;
	dx.m = Y.m;
	dx.n = Y.n;
}
__global__ void update_dX2_kernel(real* dx, real* gamma, real* sigma, int m, int n)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < m*n; k += stride) {
		int i = k / n;
		dx[k] *= gamma[i] / sigma[i];
	}
}
void update_dX2(matrix& dx, matrix& gamma, matrix& sigma)
{
	int numBlocks = (dx.n*dx.m + block_size - 1) / block_size;
	update_dX2_kernel << <numBlocks, block_size >> >(dx.data, gamma.data, sigma.data, dx.m, dx.n);
}

matrix randn(int m, int n)
{
	matrix res(m, n);
	real* tmp = (real*)malloc(sizeof(real)*m*n);
	for (int i = 0; i < m*n; i++)
		tmp[i] = normal(mt);
	cudaMemcpy(res.data, tmp, sizeof(real)*m*n, cudaMemcpyHostToDevice);
	free(tmp);
	return res;
}

matrix randn(int n)
{
	matrix res(n);
	real* tmp = (real*)malloc(sizeof(real)*n*n);
	for (int i = 0; i < n*n; i++)
		tmp[i] = normal(mt);
	cudaMemcpy(res.data, tmp, sizeof(real)*n*n, cudaMemcpyHostToDevice);
	free(tmp);
	return res;
}