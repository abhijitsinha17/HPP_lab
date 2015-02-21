#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Thread block size
#define BLOCK_SIZE 16
// Tile Width size
#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int m = numARows;
  int n = numBRows;
  int k = numBColumns;
  
  int numRows = blockIdx.y * blockDim.y + ty;
  int numColumns = blockIdx.x * blockDim.x + tx;
  float Cval = 0.0;
  
  //Loading A and B elements and doing Boundary Check
  for(int t = 0; t < (n-1)/TILE_WIDTH + 1; t++) {
  
	if((numRows < numARows) && (t*TILE_WIDTH+tx < n)) {
		ds_A[ty][tx] = A[numRows*n + t*TILE_WIDTH+tx];
	} else {
		ds_A[ty][tx] = 0.0;
	}
	
	if((numColumns < k) && (t*TILE_WIDTH+ty < n)) {
		ds_B[ty][tx] = B[(t*TILE_WIDTH+ty)*k + numColumns];
	} else {
		ds_B[ty][tx] = 0.0;
	}
	__syncthreads();
	
	for(int i = 0; i < TILE_WIDTH; i++) {
		Cval += ds_A[ty][i] * ds_B[i][tx];
	}
	__syncthreads();
  }
  
  if(numRows < m && numColumns < k) {
	C[numRows*k + numColumns] = Cval;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows*numCColumns*sizeof(float));
	
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
	
  int Size_A = numARows*numAColumns*sizeof(float);
  int Size_B = numBRows*numBColumns*sizeof(float);
  int Size_C = numCRows*numCColumns*sizeof(float);
	
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void**)&deviceA, Size_A));
  wbCheck(cudaMalloc((void**)&deviceB, Size_B));
  wbCheck(cudaMalloc((void**)&deviceC, Size_C));
	
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, Size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, Size_B, cudaMemcpyHostToDevice);
	
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1); 
  dim3 dimGrid(((numBColumns-1)/dimBlock.x)+1, ((numARows-1)/dimBlock.y)+1, 1);
	
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, Size_C, cudaMemcpyDeviceToHost);
	
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
	
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
