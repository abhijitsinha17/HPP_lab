#include <wb.h> //@@ wb include opencl.h for you

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>

//@@ OpenCL Kernel
const char *kernelSource =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  __global float *a,                       \n" \
"                       __global float *b,                       \n" \
"                       __global float *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
;

 
int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
    
	
	// Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
	
	cl_platform_id cpPlatform;       // OpenCL platform
    cl_device_id device_id;         // device ID
    cl_context clctx;               // context
    cl_command_queue clcmdqueue;    // command queue
    cl_program clpgm;               // program
    cl_kernel clkernel;             // kernel
	
	// Length of vectors
    unsigned int n = inputLength;
	
    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(float);
    size_t globalSize, localSize;
    cl_int err;
	
	// Number of work items in each local work group
    localSize = 32;
 
    // Number of total work items - localSize must be divisor
    globalSize = ceil(n/(float)localSize)*localSize;
	
	// Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 
    // Create a context  
    clctx = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue 
    clcmdqueue = clCreateCommandQueue(clctx, device_id, 0, &err);
 
    // Create the compute program from the source buffer
    clpgm = clCreateProgramWithSource(clctx, 1,
                            (const char **) & kernelSource, NULL, &err);
 
    // Build the program executable 
    clBuildProgram(clpgm, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    clkernel = clCreateKernel(clpgm, "vecAdd", &err);
	
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
    
	// Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(clctx, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(clctx, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
	err = clEnqueueWriteBuffer(clcmdqueue, d_a, CL_TRUE, 0,
                                   bytes, hostInput1, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(clcmdqueue, d_b, CL_TRUE, 0,
                                   bytes, hostInput2, 0, NULL, NULL);
 
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
   
	// Set the arguments to our compute kernel
    err  = clSetKernelArg(clkernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(clkernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(clkernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(clkernel, 3, sizeof(unsigned int), &n);
	
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

	// Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(clcmdqueue, clkernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
	// Wait for the command queue to get serviced before reading back results
	clFinish(clcmdqueue);
		
  //cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

	// Read the results from the device
    clEnqueueReadBuffer(clcmdqueue, d_c, CL_TRUE, 0,
                                bytes, hostOutput, 0, NULL, NULL );
								
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
	
	// release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(clpgm);
    clReleaseKernel(clkernel);
    clReleaseCommandQueue(clcmdqueue);
    clReleaseContext(clctx);
	
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
