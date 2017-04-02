#ifndef MATRIX_H
#define MATRIX_H
#include <cmath>
#include <cstdlib>
#include <cstdio>
typedef double real;
class matrix
{
public:
	matrix();
	matrix(int n);
	matrix(int m, int n);
	matrix(const matrix& a);
	int row_num();
	int column_num();
	matrix transpose();
	matrix& operator=(const matrix& a);
	matrix row(int k);
	matrix row(int h, int k);
	matrix column(int k);
	matrix column(int h, int k);
	void set(int k, real y);
	void set(int k, int h, real y);
	real operator()(int k);
	real operator()(int k, int h);
	friend matrix operator+(matrix &a, matrix &b);
	friend matrix operator+(matrix &a, real b);
	friend matrix operator-(matrix &a, matrix &b);
	friend matrix operator-(matrix &a, real b);
	friend matrix operator*(matrix &a, matrix &b);
	friend matrix operator*(matrix &a, real b);
	friend matrix operator*(real a, matrix &b);
	friend matrix operator/(matrix &a, real b);

	void operator+=(matrix &a);
	void operator+=(real a);
	void operator-=(matrix &a);
	void operator-=(real a);

	void operator*=(matrix &a);
	void operator*=(real a);
	void operator/=(matrix& a);
	void operator/=(real a);
	void mask(real x);

	friend void mul(matrix &a, matrix &b, matrix& res);
	friend void mul(matrix &a, real &b, matrix& res);
	friend void transposed_mul(matrix& a, matrix& b, matrix& res);

	friend matrix operator>(matrix &a, real b);
	friend matrix operator>(matrix &a, matrix &b);
	friend matrix operator>=(matrix &a, real b);
	friend matrix operator>=(matrix &a, matrix &b);
	friend matrix operator==(matrix &a, real b);
	friend matrix operator==(matrix &a, matrix &b);

	friend matrix max(real a, matrix& b);
	friend matrix max(matrix& a, real b);
	friend matrix max(matrix& a, matrix& b);
	friend int size(matrix &a, int d);
	friend matrix size(matrix& a);

	friend matrix sum(matrix& a);
	friend real norm2(matrix& a);
	friend real sum_all(matrix& a);
	friend matrix mean(matrix& a);
	friend matrix zeros(int n);
	friend matrix zeros(matrix& a);
	friend matrix zeros(int m, int n);
	friend matrix ones(int n);
	friend matrix ones(matrix& a);
	friend matrix ones(int m, int n);
	friend matrix sqrt(matrix& a);
	friend matrix dot_mul(matrix& a, matrix& b);
	friend matrix transposed_mul (matrix& a, matrix& b);
	friend matrix dot_div(matrix& a, matrix& b);

	friend real get_dx(matrix& dx,matrix& X,matrix &y);
	friend void get_dW(matrix& dW, matrix& dx, matrix& Y);
	friend void update_dX1(matrix& W, matrix& dx, matrix& Y);
	friend void update_dX2(matrix& dx, matrix& gamma, matrix& sigma);

	void print();
	void foutput(FILE* f);
	void fload(FILE* f);
	friend matrix randn(int m, int n);
	friend matrix randn(int n);
	~matrix();
protected:
	int m, n;
	real* data = NULL;
	real get(int k);
	real get(int k, int h);
};
#endif