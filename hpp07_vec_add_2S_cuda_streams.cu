#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int idx  = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx  < len) {
		out[idx ] = in1[idx] + in2[idx];
	}
}


int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;

	//Host pinned memory
	float *h_Ap, *h_Bp, *h_Cp;

	//Device memory for each stream
	float *d_A0, *d_B0, *d_C0;
	float *d_A1, *d_B1, *d_C1;
	
	
	//CUDA Stream: the “chunked” computation and the overlap of memory copies with kernel execution.
	//get stream 1 to copy its input buffers to the GPU while stream 0 is executing its kernel. 
	//Then stream 1 will execute its kernel while stream 0 copies its results to the host.
	//Stream 1 will then copy its results to the host while stream 0 begins executing its kernel on the next chunk of data
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
 
	h_Ap = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLengt);
    h_Bp = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    h_Cp = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
	wbLog(TRACE, "The input length is ", inputLength);
		
	int size = inputLength * sizeof(float);
	int SegSize = inputLength/4;
	
	// allocate page-locked memory, used to stream
	wbCheck(cudaHostAlloc((void **) &h_Ap, size, cudaHostAllocDefault));
	wbCheck(cudaHostAlloc((void **) &h_Bp, size, cudaHostAllocDefault));
	wbCheck(cudaHostAlloc((void **) &h_Cp, size, cudaHostAllocDefault));
	
	//@@ Allocate GPU device memory here
	wbCheck(cudaMalloc((void **) &d_A0, size));
	wbCheck(cudaMalloc((void **) &d_B0, size));
	wbCheck(cudaMalloc((void **) &d_C0, size));
	
	wbCheck(cudaMalloc((void **) &d_A1, size));
	wbCheck(cudaMalloc((void **) &d_B1, size));
	wbCheck(cudaMalloc((void **) &d_C1, size));

	
	//@@ Initialize the grid and block dimensions here
	dim3 DimGrid(((inputLength-1)/256 + 1) , 1 , 1);
	dim3 DimBlock(256 , 1, 1);
	
	// now loop over full data, in chunks
	for (int i=0; i<size; i+=SegSize*2) 
	{	
		//Copy data asynchronously to device using streams and launch kernel. 
		cudaMemcpyAsync(d_A0, h_Ap+i, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, h_Bp+i, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
		
		cudaMemcpyAsync(d_A1, h_Ap+i+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1);
		cudaMemcpyAsync(d_B1, h_Bp+i+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1);
		
		//Launching kernel to avoid overlap between Copy Engine and Kernel Engine in PCIe
		//This kernel launch is asynchronous
		vecAdd<<<DimGrid, DimBlock, 0, stream0>>>(d_A0, d_B0, d_C0,inputLength);
		vecAdd<<<DimGrid, DimBlock, 0, stream1>>>(d_A1, d_B1, d_C1,inputLength);
	
		//Synch the streams before copying data back.
		cudaDeviceSynchronize();
		
		//Copy data back to host.
		// you are only allowed to schedule asynchronous copies to or from page-locked memory. 
		cudaMemcpyAsync(h_Cp+i, d_C0, SegSize*sizeof(float),cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(h_Cp+i+SegSize, d_C1, SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream1);

		wbLog(TRACE, "on addition is ", *h_Cp);
	}
	
	wbTime_start(GPU, "Freeing GPU Memory");
	
    //@@ Free the GPU memory here
	// cleanup the streams and memory
	cudaFreeHost(h_Ap);
	cudaFreeHost(h_Bp);
	cudaFreeHost(h_Cp);
		
	cudaFree(d_A0);
	cudaFree(d_B0);
	cudaFree(d_C0);
	
	cudaFree(d_A1);
	cudaFree(d_B1);
	cudaFree(d_C1);

    wbTime_stop(GPU, "Freeing GPU Memory");
	
	cudaStreamDestroy( stream0 );
	cudaStreamDestroy( stream1 );
	
	wbSolution(args, hostOutput, inputLength);

    return 0;
}

