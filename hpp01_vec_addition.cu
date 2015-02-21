CPU: Latency cores
Powerful ALU's, large cache(reduce memory access latency), few registers, few SIMD units, sophisticated control logic(reduce branch latency by prediction & reduce data latency).

GPU: Throughput cores
Small caches(to boost throughput)
Simple COntrol: No BP, No data forwarding
Energy efficient ALU's: Many long latency but heavily pipelined ALU's for high throughput.
So it needs a massive number of threads to tolerate latencies.

We need Applications that use both CPU and GPU:
CPU for sequential parts for latency matters
GPU for parallel parts where throughput wins.
----------------------------------------------------------------
Scalability and Portability in parallel computing:

Scalability:
A sw which run on coreA should also run on coreA++
same sw can run on more of the coreA's

Potability:
A same sw should be able to run on CoreA CoreB and CoreC.(different hw types and different hw vendors.)
eg: x86 vs ARM(diff instruction sets)
	latency oriented CPU vs throughput oriented GPU
	VLIW vs SIMD vs Multi-threading
	Shared memory model vs distributed memory model.
	
-----------------------------------------------------------------
Introduction to CUDA:

Data Parallelism.
Heterogeneous Host+Device application C Program:
- Serial parts in "Host" & 
- Parallel parts in "Device" ~ throughput oriented GPU

Parallel kernel(device): KernelA <<<nBlk, nTid>>>(args);

ISA:Instruction Set Architecture: A contract between h/w and s/w. a set of instructions that a h/w can execute.
Whenever a code is compiled its converted to a set of instructions in memory(Instruction Register) which can be executed by h/w.

Von-Neumann Architecture : Modern processor design.

Cuda Thread:
A cuda thread is an "abstracted" or "virtualized" form of Von_Neumann processor

CUDA Kernel:
- cuda kernel is executed in Grid(array) of threads.
- an index is calculated for each thread & it uses it to compute memory addresses & make control decisions.

Index: [i = blockIdx.x * blockDim.x + threadIdx.x;]

Block 0 will have i value= 0 to 255;
Block 1 will have i value= 256 to 511;
Block 2 will have i value= 512 to ;

- Threads within a block cooperate via Shared Memory, Atomic operations and barrier synchronization.
- Threads in different blocks do not interact.
- Block index and thread index can be in 1D, 2D and 3D.

------------------------------------------------------------------
API function in CUDA host code:
1. Device memory allocation
2. Host-device data transfer

Vector addition code
-Host can transfer data to-fro from global device memory.

cudaMalloc():
allocates objects in device global memory.
args: address of the "pointer to the allocated object" & Size in bytes.



cudaFree():
Frees object from device global memory.
- pointer to "freed object".

cudaMemcpy():
memory data transfer.
args: pD, pS, NumBytes, type/direction of transfer(defined in cuda.h).

Always check for error conditions:
<use it in lab assignments>
cudaError_t var_err = cudaMalloc((void**)&d_A, size);
if(err!=cudaSuccess){
	CudaGetErrorString(var_err),__FILE__, __LINE__);
	exit(EXIT_FAILURE);
}
Generally error happens when there is not enough device memory.

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
	
-----------------------------------------------------------------
KErnel based SPMD programming:
-declaration
-built in variables
-thread index to data index mapping.

Device Code:
__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<n) //boundary check for threads.
	{
		C[i] = A[i] + B[i];
	}
}

Host Code:
int vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
	//d_A, d_B, d_C allocations in device memory: cudaMalloc
	//cudaMemcpy from host to device
	//Run ceil (n/256.0): float division, blocks of 256 threads each.
Kernel Invocation: <<<No. of blocks, no. of threads>>>
	vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
	//copy device memory back to host
	//free device memory: cudaFree. 
}	

More than 1 dimension:
int vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
//since we are using one dimensional grid, so we just need to initialize the x value.
	dim3 DimGrid((n-1)/256 + 1, 1, 1); // no.of blocks in grid
	dim3 DimBlock(256,1, 1); // no. of threads in block.
	vecAddKernel<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, n);
}

3 types of function in CUDA programming:
1. __device__ : exec in device, called by device
2. __global__ : exec in device, called by Host (declares kernel) must return "void".
3. __host__   : exec in host, called by host

Note: 1 and 3 can be used for functions getting compiled and used on both device and host.

Compilation of Cuda program: NVCC compiler
1. Host C compiler(gcc,icc)/ Linker
2. Device code PTX binary format(like java bikel), just-in-time compiler(take PTX and generate real ISA for the device).

-----------------------------------------------------------------
Multidimensional Grids:
- Multi dimensional block and thread indices
- Mapping block/thread indices to data indices

In modern computer - all data is stored in linear address space. 
- C/C++ has a Row-Major Layout of memory. 
i.e. all the elements in the row will have their position preserved.
- To access an element using a linear address we can generate the address by multiplying the row index by the width of the array plus the column index.
[ Index = Row * Width + Col ]

For m x n picture:
m: pixels in y dim
n: pixels in x dim

Eg:  
image 600x800 (y * x)
m: y 600
n: x 800 

Note: because of boundary check...Not all threads in a Block will follow the same. Performance implications.

Row indices: take y
Column indices: take x