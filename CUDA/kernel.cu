
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrix.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "test.h"
#include "distribution_classification.h"
int main()
{
	//RMSProp	44.7
	//SGD		142.365
	//Adma		7.76
	//AdaGrad	
	test(1024, 256, 1e-3,1e-3);
	//distribution_classification(2560, 256, 1e-3, 1e-3);
	system("pause");
	return 0;
}