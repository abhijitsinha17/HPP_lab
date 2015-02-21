#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define TILE_WIDTH 16
#define BLOCK_WIDTH (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE
__global__ void convolution_kernel(float *I, float *P,  const float* __restrict__ M,
							int channels, int width, int height) { 
	// Allocating shared memory 
	__shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];
	int k;
	for (k = 0; k < channels; k++) {
		// Loading 1st set of data
		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
		int ty = dest / BLOCK_WIDTH;
		int tx = dest % BLOCK_WIDTH;
		int row_i = blockIdx.y * TILE_WIDTH + ty - Mask_radius;
		int col_i = blockIdx.x * TILE_WIDTH + tx - Mask_radius;
		if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
			N_ds[ty][tx] = I[(row_i * width + col_i) * channels + k];
		}
		else {
			N_ds[ty][tx] = 0.0;
		}
	
		// Loading 2nd set of data
		dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
		ty = dest / BLOCK_WIDTH;
		tx = dest % BLOCK_WIDTH;
		row_i = blockIdx.y * TILE_WIDTH + ty - Mask_radius;
		col_i = blockIdx.x * TILE_WIDTH + tx - Mask_radius;
		if (ty < BLOCK_WIDTH) {
			if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
				N_ds[ty][tx] = I[(row_i * width + col_i) * channels + k];
			}
			else {
				N_ds[ty][tx] = 0.0;
			}
		}
		__syncthreads();
		
		float accum = 0.0f;
		int y;
		int x;
		for (y = 0; y < Mask_width; y++) {
			for (x = 0; x < Mask_width; x++) {
				accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
			}
		}	
		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
		if (y < height && x < width) {
			P[(y * width + x) * channels + k] = clamp(accum);
		}
		__syncthreads();
	} 
		
}	

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	dim3 dimGrid((((imageWidth-1.0)/TILE_WIDTH) + 1.0), (((imageHeight-1.0)/TILE_WIDTH) + 1.0), 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	convolution_kernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData,
										imageChannels, imageWidth, imageHeight);
										
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
