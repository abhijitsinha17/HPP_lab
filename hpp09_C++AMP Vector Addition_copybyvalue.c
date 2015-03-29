// Copy by value

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
	//capture variables by value
  	array_view<float,1> AV(n,hostInput1), BV(n,hostInput2);
	array_view<float,1> CV(n,hostOutput);

	parallel_for_each(CV.get_extent(), [=](index<1> i)

	//It has to run on a device
	restrict(amp) {
		CV[i] = AV[i] + BV[i];
	});
	CV.synchronize(); 
  
  wbSolution(args, hostOutput, inputLength);

  return 0;
}
