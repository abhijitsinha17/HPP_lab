#include <wb.h>
#include <amp.h>

using namespace concurrency;

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  //@@ Insert C++AMP code here
  unsigned int n = inputLength;

	//capture variables by reference
	array<float,1> AA(n), BA(n);
	array<float,1> CA(n);
	copy(hostInput1, AA);
	copy(hostInput2, BA);
	
	parallel_for_each(CA.get_extent(), [&AA, &BA, &CA](index<1> i)
	//It has to run on a device
	restrict(amp) {
	CA[i] = AA[i] + BA[i];
	});
	//exclusive copy to output
	copy(CA, hostOutput);
  
  wbSolution(args, hostOutput, inputLength);

  return 0;
}
