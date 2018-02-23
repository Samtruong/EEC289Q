#include "ParallelGreedy.cu"
#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int** color = (int**) malloc(1*sizeof(int*));
  GraphColoringGPU(argv[1],color);
  for(int i=0;i<16;i++)
  {cout<<color[i]<<endl}
  return 0;
}
