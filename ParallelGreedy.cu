#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

using namespace std;

void SerialThrust(int* h_graph, int* dimension, int V)
{
  for(int row = 0; row < V; row++)
  {
    thrust::exclusive_scan(&h_graph[V*row],&h_graph[V*row + V],&h_graph[V*row]);
    dimension[row] = h_graph[V*row + V -1]+1;
  }
}

__global__ void ParallelThrust(int* h_graph, int* dimension, int V)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; index < V*V; index += stride)
  {
    thrust::exclusive_scan(thrust::device,&h_graph[i*V],&h_graph[i*V+V],&h_graph[i*V]);
  }
}
//================================Utility Functions=======================================

//Load raw .co data
void getDimension(const char filename[], int* V)
{
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
   }

   int num_rows;

   while (getline(infile, line))
   {
      istringstream iss(line);
      string s;
      iss >> s;
      if (s == "p") {
         iss >> s; // read string "edge"
         iss >> num_rows;
         *V = num_rows;
         break;
      }
   }
   infile.close();
}

void ReadColFile(const char filename[], int* graph, int V)
{
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
   }

   while (getline(infile, line)) {
      istringstream iss(line);
      string s;
      int node1, node2;
      iss >> s;
      if (s != "e")
         continue;

      iss >> node1 >> node2;

      // Assume node numbering starts at 1
      (graph)[(node1 - 1) * V + (node2 - 1)] = 1;
      (graph)[(node2 - 1) * V + (node1 - 1)] = 1;
   }
   infile.close();
}

//print graph Matrix
void PrintMatrix(int* matrix, int M, int N) {
   for (int row=0; row<M; row++)
   {
      for(int columns=0; columns<N; columns++)
      {
         printf("%i", matrix[row * N + columns]);
      }
      printf("\n");
   }
}


//===================================Main================================================

int main(int argc, char* argv[])
{
   int* h_graph;
   int V;
   //int* color;
   if (string(argv[1]).find(".col") != string::npos)
   {
     getDimension(argv[1], &V);
     cudaMallocManaged(&h_graph,sizeof(int)*V*V);
     ReadColFile(argv[1],h_graph,V);
   }
   //else if (string(argv[1]).find(".mm") != string::npos)
      //ReadMMFile(argv[1], &graph, &V);
   else
      return -1;


   int dimension[V];
   ParallelThrust<<<V,V>>>(h_graph,dimension,V);
   cudaDeviceSynchronize();
   cout<<"Scan Graph"<<endl;
   PrintMatrix(h_graph,V,V);
   // cout<<"dimension"<<endl;
   // PrintMatrix(dimension,1,V);
   cudaFree(h_graph);
   return 0;
}
