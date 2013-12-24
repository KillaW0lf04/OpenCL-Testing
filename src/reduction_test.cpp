#include <stdio.h>
#include <math.h>
#include <string>

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "util.hpp"

#include <CL/cl.h>

using namespace std;

int main(int argc, char *argv[])  {

	cl::Context context(CL_DEVICE_TYPE_GPU);
	cl::Program program(context, util::loadProgram("oclReduction_kernel.cl"), false);
	cl::CommandQueue queue(context);

	vector<cl::Device> devices =  context.getInfo<CL_CONTEXT_DEVICES>();

	try {
		program.build();

		// Works for any array which is a power of 2
		// If you array is not a power of 2, simply pretend it is and append 0s at the end for the reduction
		// You can swap between levels of optimisation by choosing reduce0 - reduce6

		// PASS: Reduction0
		// PASS: Reduction1
		// PASS: Reduction2
		// PASS: Reduction3
		// PASS: Reduction4
		// FAIL: Reduction5
		// FAIL: Reduction6
		auto d_reduction = cl::make_kernel<cl::Buffer, cl::Buffer, int, cl::LocalSpaceArg>(program, "reduce4");

		const int n = 300 * 200;
		//int n = pow(2, 10);
		//int n = 32 * 100;

		// This parameter needs to be tweaked for performance
		// TODO: Bug when setting workgroup size = 32
		const int workgroup_size  = 64;
		const int dim =ceil(n / (float) workgroup_size) * workgroup_size;

		// Result size needs to be as large as the number of workgroups
		vector<float> result(dim / workgroup_size);

		vector<float> numbers(n);
		for(int i=0; i<numbers.size(); i++)
			numbers[i] = 1.0;
		for(int i=0; i<result.size(); i++)
			result[i] = 0.0;

		int local_size = sizeof(int) * workgroup_size;;

		printf("Performing Reduction....\n");
		printf("Workgroup Size = %d\n", workgroup_size);
		printf("N = %d\n", n);
		printf("DIM = %d\n", dim);
		printf("Results size = %.2f\n", dim / (float) workgroup_size);
		printf("Local Memory size (per WG) = %d bytes\n", local_size);

		if(local_size > 48 * 1024)
			printf("WARNING: Local Memory size is over the expected limit!!!!!\n");

		// Local Space = number of work items in a group
		cl::LocalSpaceArg SWrk = cl::Local(local_size);
		cl::Buffer d_numbers(context, begin(numbers), end(numbers), true);
		cl::Buffer d_result(context, begin(result), end(result), false);

		util::Timer timer;

		printf ("Launching Kernel\n");

		// Make sure your workgroup size is inside NDRange to prevent bugs!
		d_reduction(cl::EnqueueArgs(queue, cl::NDRange(dim), workgroup_size), d_numbers, d_result, numbers.size(), SWrk);

		cl::copy(queue, d_result, begin(result), end(result));
		printf("Time Taken = %ld ns\n", timer.getTimeNanoseconds());

		float total = 0;
		for(int i=0; i<result.size();i++) {
			printf("%.2f, ", result[i]);
			total += result[i];
		}
		printf("\n");

		//int expected = n * (n - 1);
		//expected /= 2.0;

		float expected = n;

		printf("Total = %.2f\n", total);
		printf("Expected =  %.2f\n", expected);

		if(total == expected)
			printf("Finished Successfully!\n");
		else
			printf("Finished with Errors! :(\n");


	} catch(cl::Error err) {
		if(err.err() == CL_BUILD_PROGRAM_FAILURE) {
			string build_error = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
			std::cout << "Error compiling Kernel code: " << build_error << endl;
		}
		else {
			std::cout << "OpenCL Error: " << err.what() << " returned " << err.err() << std::endl;
			std::cout << "Check cl.h for error codes." << std::endl;
		}
		exit(EXIT_FAILURE);
	}
}
