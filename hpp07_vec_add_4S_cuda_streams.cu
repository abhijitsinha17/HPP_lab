#include	<wb.h>
#include	<wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
		
__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len) {
		out[i] = in1[i] + in2[i];
	}
}


int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;

    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
	
	//Device memory for each stream
	float *d_A0, *d_B0, *d_C0;
	float *d_A1, *d_B1, *d_C1;
	float *d_A2, *d_B2, *d_C2;
	float *d_A3, *d_B3, *d_C3;
	
	
	//CUDA Stream: the “chunked” computation and the overlap of memory copies with kernel execution.
	//get stream 1 to copy its input buffers to the GPU while stream 0 is executing its kernel. 
	//Then stream 1 will execute its kernel while stream 0 copies its results to the host.
	//Stream 1 will then copy its results to the host while stream 0 begins executing its kernel on the next chunk of data
	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
	wbLog(TRACE, "The input length is ", inputLength);

		
	int size = inputLength * sizeof(float);
	int SegSize = inputLength/4;
	
	//@@ Allocate GPU device memory here
	wbCheck(cudaMalloc((void **) &d_A0, size));
	wbCheck(cudaMalloc((void **) &d_B0, size));
	wbCheck(cudaMalloc((void **) &d_C0, size));
	
	wbCheck(cudaMalloc((void **) &d_A1, size));
	wbCheck(cudaMalloc((void **) &d_B1, size));
	wbCheck(cudaMalloc((void **) &d_C1, size));
	
	wbCheck(cudaMalloc((void **) &d_A2, size));
	wbCheck(cudaMalloc((void **) &d_B2, size));
	wbCheck(cudaMalloc((void **) &d_C2, size));
	
	wbCheck(cudaMalloc((void **) &d_A3, size));
	wbCheck(cudaMalloc((void **) &d_B3, size));
	wbCheck(cudaMalloc((void **) &d_C3, size));
	
	//@@ Initialize the grid and block dimensions here
	dim3 DimGrid(((inputLength-1)/256 + 1) , 1 , 1);
	dim3 DimBlock(256 , 1, 1);
	
	// now loop over full data, in chunks
	for (int i=0; i<size; i+=SegSize*4) 
	{	
		//Copy data asynchronously to device using streams and launch kernel. 
		cudaMemcpyAsync(d_A0, hostInput1+i, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, hostInput2+i, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
		
		cudaMemcpyAsync(d_A1, hostInput1+i+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1);
		cudaMemcpyAsync(d_B1, hostInput2+i+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1);
		
		cudaMemcpyAsync(d_A2, hostInput1+i+SegSize+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(d_B2, hostInput2+i+SegSize+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream2);
		
		cudaMemcpyAsync(d_A3, hostInput1+i+SegSize+SegSize+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(d_B3, hostInput2+i+SegSize+SegSize+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream3);
		
		//Launching kernel to avoid overlap between Copy Engine and Kernel Engine in PCIe
		//This kernel launch is asynchronous
		vecAdd<<<DimGrid, DimBlock, 0, stream0>>>(d_A0, d_B0, d_C0,inputLength);
		vecAdd<<<DimGrid, DimBlock, 0, stream1>>>(d_A1, d_B1, d_C1,inputLength);
		vecAdd<<<DimGrid, DimBlock, 0, stream2>>>(d_A2, d_B2, d_C2,inputLength);
		vecAdd<<<DimGrid, DimBlock, 0, stream3>>>(d_A3, d_B3, d_C3,inputLength);
				
		//Synch the streams before copying data back.
		cudaDeviceSynchronize();
		
		//Copy data back to host.
		// you are only allowed to schedule asynchronous copies to or from page-locked memory. 
		cudaMemcpyAsync(hostOutput+i, d_C0, SegSize*sizeof(float),cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(hostOutput+i+SegSize, d_C1, SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream1);
		cudaMemcpyAsync(hostOutput+i+SegSize+SegSize, d_C2, SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream2);
		cudaMemcpyAsync(hostOutput+i+SegSize+SegSize+SegSize, d_C3, SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream3);

		wbLog(TRACE, "on addition is ", *hostOutput);
	}
	
	wbSolution(args, hostOutput, inputLength);
	
    //@@ Free the GPU memory here
	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(d_A0);
	cudaFree(d_B0);
	cudaFree(d_C0);
	
	cudaFree(d_A1);
	cudaFree(d_B1);
	cudaFree(d_C1);
	
	cudaFree(d_A2);
	cudaFree(d_B2);
	cudaFree(d_C2);
	
	cudaFree(d_A3);
	cudaFree(d_B3);
	cudaFree(d_C3);	
    wbTime_stop(GPU, "Freeing GPU Memory");
	
	cudaStreamDestroy( stream0 );
	cudaStreamDestroy( stream1 );
	cudaStreamDestroy( stream2 );
	cudaStreamDestroy( stream3 );
	
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

