#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>

using namespace std;


__global__ 
void bin(bool * graph, int V, int* color)
{


}

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


int main(int argc, char* argv[])
{
   bool* graph;
   int V;
   int* color;

   if (string(argv[1]).find(".col") != string::npos)
      ReadColFile(argv[1], &graph, &V);
   else if (string(argv[1]).find(".mm") != string::npos)
      ReadMMFile(argv[1], &graph, &V);
   else
      return -1;

 //  GraphColoring(graph, V, &color);
 //  printf("Brute-foce coloring found solution with %d colors\n", CountColors(V, color));
 //  printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));

   GreedyColoring(graph, V, &color);
   printf("Greedy coloring found solution with %d colors\n", CountColors(V, color));
   printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));

	
   return 0;
}
