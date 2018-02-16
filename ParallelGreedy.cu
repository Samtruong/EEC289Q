#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>

using namespace std;
__global__ // number of threads must be V/2
void BlockScan(bool* h_row, int V)
{
  extern __share__ int d_row[];
  d_row[2*threadIdx.x] = h_row[2*threadIdx.x];
  d_row[2*threadIdx.x + 1] = h_row[2*threadIdx.x + 1];
  offset = 1;
//WALK TO ROOT
  for(int layer = V/2; layer > 0; layer/=2)
  {
    __syncthread();
    if(thread.x < layer)
    {
      int leftHandSide = offset*(2*threadIdx.x+1)-1;
      int rightHandSide = offset*(2*threadIdx.x+2)-1;

      d_row[rightHandSide] += d_row[leftHandSide];
    }
    offset*=2;
  }
//WALK FROM ROOT
  if (threadIdx.x == 0){d_row[V-1] = 0;}
  for(int layer = 1; layer < V; layer*=2)
  {
    offset /= 2;
    __syncthread();
    if(threadIdx < layer)
    {
      int leftHandSide = offset*(2*threadIdx.x + 1)-1;
      int rightHandSide = offset*(2*threadIdx.x +2)-1;
      int holder = d_row[leftHandSide];
      d_row[leftHandSide] = d_row[rightHandSide];
      d_row[rightHandSide] += holder;
    }
  }
  __syncthread();

//RETURN RESULT
  h_row[2*threadIdx.x] = d_row[2*threadIdx.x];
  h_row[2*threadIdx.x + 1] = d_row[2*threadIdx.x + 1];
}

//================================Utility Functions=======================================

//Load raw .co data
void ReadColFile(const char filename[], bool** graph, int* V)
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
         *graph = new bool[num_rows * num_rows];
         memset(*graph, 0, num_rows * num_rows * sizeof(bool));
         continue;
      } else if (s != "e")
         continue;

      iss >> node1 >> node2;

      // Assume node numbering starts at 1
      (*graph)[(node1 - 1) * num_rows + (node2 - 1)] = true;
      (*graph)[(node2 - 1) * num_rows + (node1 - 1)] = true;
   }
   infile.close();
}

//print graph Matrix
void PrintMatrix(bool* matrix, int M, int N) {
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
   bool* graph;
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
   PrintMatrix(graph,V,V);
   return 0;
}
