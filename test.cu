#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>



__global__
void makeBins(int * coloredGraph, int numV, int numC, std::queue<int>* bins)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index ; i < numV; i+= stride)
  {
    bins[coloredGraph[i]].push(i);
  }
}

int main(int argc, char const *argv[])
{
  const int numV = 10;
  const int numC = 5;
  int coloredGraph[numV];

  for(int i = 0; i <  numV; i++)
  {
    coloredGraph[i] = rand() % 1000 + 1;
  }


  std::queue<int> * bins;
  cudaMallocManaged(& bins, numC*numV*sizeof(int));
  
  makeBins<<<1,1>>>(coloredGraph, numV, numC, bins);
  return 0;
}
