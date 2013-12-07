
// Performs a matrix multiplication on a NxN matrix
__kernel void matrixmul(const int N, __global float *A, __global float *B, __global float *C) 
{
	int i = get_global_id(0);
	int j, k;

	float BWork[1024];

	for(k=0; k<N; ++k) {
		BWork[k] = B[i * N + k];
	}

	for(j=0; j<N; ++j) {

		float temp = 0.0f;

		for(k=0; k<N; ++k) {
			temp += BWork[k] * C[k * N + j];
		}

		A[i * N + j] = temp;
	}
}