#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <queue>
#include <device_vector.h>

  //cudaMallocManaged(& bins, numC*numV*sizeof(int));

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

__global__
void reduceBins(int * coloredGraph, int numV, int numC, std::queue<int> * bins, int threshold)
{
	int binIndex = blockIdx.x % numC;
	std::queue<int> bin = bins[binIndex];
	int iteration = 0;
	while (iteration < threshold)
	{	
		int currentVertex = queue.pop();
			

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


  
  std::queue<int>  bins [numC];
  makeBins<<<1,1>>>(coloredGraph, numV, numC, bins);
  
//cudaFree(bins);  
return 0;
}
