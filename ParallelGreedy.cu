#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <thrust/scan.h>

using namespace std;
// int index = threadIdx.x + blockIdx.x * blockDim.x;
// int stride = blockDim.x * gridDim.x;
// for(int i = index ; i < V ; i+=stride)


// __global__ // number of threads must be V/2
// void BlockScan(bool* h_row, int V)
// {
//   extern __share__ int d_row[];
//   int ID = threadIdx.x;
//   int offset = 1;// helps point to the next data element in next step
//
//   //with bank conflict:
//   // d_row[2*ID] = h_row[2*threadIdx.x];
//   // d_row[2*threadIdx.x + 1] = h_row[2*threadIdx.x + 1];
//
//   //without bank conflict: Load in shared memory
//   d_row[ID + blockDim.x] = h_row[ID + blockDim.x];
//   d_row[ID] = h_row[ID];
//
// //============================WALK TO ROOT======================================
// // walk up until you are root
//   __syncthread();
//   for(int layer = V/2; layer > 0; layer/=2)
//   {
//     if(ID < layer) // number of threads gets halfed every step
//     {
//       int leftHandSide = offset*(2*ID+1)-1;
//       int rightHandSide = offset*(2*ID+2)-1;
//       //produce partial sum:
//       d_row[rightHandSide] += d_row[leftHandSide];
//     }
//     offset*=2;
//     __syncthread();
//   }
// //end for loop
// //===========================WALK FROM ROOT=====================================
// //walk down until you are at Vth layer
//   if (ID == 0){d_row[V-1] = 0;} // replace the last element with 0
//   for(int layer = 1; layer < V; layer*=2)
//   {
//     offset /= 2;
//     __syncthread();
//     if(ID < layer)
//     {
//       int leftHandSide = offset*(2*ID + 1)-1;
//       int rightHandSide = offset*(2*ID +2)-1;
//       int holder = d_row[leftHandSide]; //copy the right element
//       d_row[leftHandSide] = d_row[rightHandSide];
//       //produce partial sum:
//       d_row[rightHandSide] += holder;
//     }
//   }
// //end for loop
//   __syncthread();
//
// //RETURN RESULT
//
// //with bank conflict
//   // h_row[2*ID] = d_row[2*ID];
//   // h_row[2*ID + 1] = d_row[2*ID + 1];
//
// //without bank conflict, loading back to global memory:
//   h_row[ID] = d_row[ID];
//   h_row[ID + blockDim.x] = d_row[ID + blockDim.x];
// }
__global__ RandomizedParallelGreedy(int** graph, int** solutions)
{
  
}
void SerialThrust(int* h_graph, int V)
{
  for(int row = 0; row < V; row++)
  {
    thrust::exclusive_scan(&h_graph[V*row],&h_graph[V*row + V],&h_graph[V*row]);
  }
}

//================================Utility Functions=======================================

//Load raw .co data
void ReadColFile(const char filename[], int** graph, int* V)
{
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
   }

   int num_rows, num_edges;

   while (getline(infile, line)) {
      istringstream iss(line);
      string s;
      int node1, node2;
      iss >> s;
      if (s == "p") {
         iss >> s; // read string "edge"
         iss >> num_rows;
         iss >> num_edges;
         *V = num_rows;
         *graph = new int[num_rows * num_rows];
         memset(*graph, 0, num_rows * num_rows * sizeof(int));
         continue;
      } else if (s != "e")
         continue;

      iss >> node1 >> node2;

      // Assume node numbering starts at 1
      (*graph)[(node1 - 1) * num_rows + (node2 - 1)] = 1;
      (*graph)[(node2 - 1) * num_rows + (node1 - 1)] = 1;
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
   int* graph;
   int V;
   //int* color;

   if (string(argv[1]).find(".col") != string::npos)
      ReadColFile(argv[1], &graph, &V);
   //else if (string(argv[1]).find(".mm") != string::npos)
      //ReadMMFile(argv[1], &graph, &V);
   else
      return -1;

 //  GraphColoring(graph, V, &color);
 //  printf("Brute-foce coloring found solution with %d colors\n", CountColors(V, color));
 //  printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));

   // GreedyColoring(graph, V, &color);
   // printf("Greedy coloring found solution with %d colors\n", CountColors(V, color));
   // printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));
   // cout<<"Original Graph"<<endl;
   // PrintMatrix(graph,V,V);
   SerialThrust(graph,V);
   // cout<<"Scan Graph"<<endl;
   // PrintMatrix(graph,V,V);
   return 0;
}
