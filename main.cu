#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here
/*
__device__
bool between(int element, int a, int b, int* dx, int* dy){
  int min_x = min(dx[a], dx[b]);
  int max_x = max(dx[a], dx[b]);
  int min_y = min(dy[a], dy[b]);
  int max_y = max(dy[a], dy[b]);

  if(dx[element] >= min_x && dx[element] <= max_x)
    if(dy[element] >= min_y && dy[element] <= max_y)
      return true;
  return false;
}

__device__
bool same_line(int target, int source, int dest, int* dx, int* dy){
  float m = (dy[dest] - dy[source])/(dx[dest] - dx[source]);
  float b = dy[dest] - m * dx[dest];
  //accuracy issue for approximation
  if(dy[target] == m * dx[target] + b) return true;
  return false;
}

//dynamic parallelism
__global__
void calcTarget(int round, int source, int target, int* dx, int* dy, int* dhp, int* dist){
  int tid = threadIdx.x;
  if(tid != source){
    if(dhp[tid] > 0){
      int slope_x = (dx[target] - dx[source])/(dx[tid] - dx[source]);
      int slope_y = (dy[target] - dy[source])/(dy[tid] - dy[source]);
      //ensure those ratios are integers
      if(slope_x == slope_y && slope_x > 0)
      dist[tid] = slope_x;
    }
  }
}

__device__
int calcTarget(int source, int target, int* dx, int* dy, int* dhp, int T){
  float dist = INT_MAX;
  int index = -1;

  for(int tid = 0; tid < T; tid++){
    if(tid != source
       && (between(tid, source, target, dx, dy) || between(target, source, tid, dx, dy))
       && same_line(tid, source, target, dx, dy)){

        if(dhp[tid] > 0){
          /*
          int slope_x = (dx[target] - dx[source])/(dx[tid] - dx[source]);
          int slope_y = (dy[target] - dy[source])/(dy[tid] - dy[source]);
          //ensure those ratios are integers
          if(slope_x == slope_y && slope_x > 0 && dist > slope_x){
            dist = slope_x;
            index = tid;
          }
          float temp = distance(source, tid, dx, dy);
          if(dist > temp)
          {
            dist = temp;
            index = tid;
          }

      }
    }
  }
  return index;
}
*/

__device__
long long int distance(int a, int b, int* dx, int* dy){
  return (pow((dx[a] - dx[b]),2) + pow((dy[a] - dy[b]),2));
}

__device__
bool same_direction(int source, int dest, int target, int* dx, int* dy){
  long long int x_dest = dx[dest] - dx[source];
  long long int y_dest = dy[dest] - dy[source];

  long long int x_target = dx[target] - dx[source];
  long long int y_target = dy[target] - dy[source];

  if(x_dest * y_target == y_dest * x_target &&
  x_dest * x_target >= 0 && y_dest * y_target >= 0)
  return true;
  return false;
}

__device__
int calcTarget(int source, int dest, int* dx, int* dy, int* dhp, int T){
  int index = -1;
  long long int dist = LLONG_MAX;

  for(int target = 0; target < T; target++){
    if(target != source && dhp[target] > 0){
      //target is in the same direction
      if(same_direction(source, dest, target, dx, dy)){
        long long int d = distance(source, target, dx, dy);
        if(dist > d){
          dist = d;
          index = target;
        }
      }
    }
  }
  return index;
}
__global__
void calculateFinalScore
(int* dx, int* dy, int* dscore, int* dhp, int *active){
  int current_round = 0;
  int tid = threadIdx.x;
  int T = blockDim.x;

  //__syncthreads();
  // printf("active %d dhp %d dscore %d\n", *active, dhp[tid], dscore[tid]);

  while((*active) > 1 && dhp[tid] > 0){ //tank tid is able to participate in current_round
    current_round++;
    // printf("tid %d current_round %d active %d\n", tid, current_round, *active);
    int target = (tid + current_round) % T;
    if(target != tid){

      //int *dist;
      //cudaMalloc(&dist, T * sizeof(int));
      //cudaMemset(dist, INT_MAX, T * sizeof(int));

      //calcTarget<<<1, T>>>(current_round, tid, target, dx, dy, dhp, dist);
      // printf("target of %d before %d: \n", tid, target);

      target = calcTarget(tid, target, dx, dy, dhp, T);
      __syncthreads();
      // printf("target of %d after %d: \n", tid, target);

      //cudaDeviceSynchronize();
      if(target != -1)
      {
        atomicAdd(&dhp[target], -1);
        atomicAdd(&dscore[tid], 1);
      }
      // printf("tid %d target %d dscore %d\n", tid, target, dscore[tid]);
      __syncthreads();
      if(dhp[tid] <= 0) atomicAdd(active, -1);
      __syncthreads();
    }

  }

}

//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;


    FILE *inputfilepointer;

    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
    //printf("initialized M: %d, N: %d, T: %d, H: %d\n", M, N, T, H);

    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *dx, *dy, *dscore, *dhp;
    int *active;
    int *hp = (int*)malloc(T * sizeof (int));
    for(int i = 0; i < T; i++) hp[i] = H;

    cudaMalloc(&dx, T * sizeof (int));
    cudaMalloc(&dy, T * sizeof (int));
    cudaMalloc(&dscore, T * sizeof (int));
    cudaMalloc(&dhp, T * sizeof(int));
    cudaMalloc(&active, sizeof(int));
    cudaMemcpy(dx, xcoord, T * sizeof(int) ,cudaMemcpyHostToDevice);
    cudaMemcpy(dy, ycoord, T * sizeof(int) ,cudaMemcpyHostToDevice);
    cudaMemcpy(dhp, hp, T * sizeof(int) ,cudaMemcpyHostToDevice);
    cudaMemset(dscore, 0, T * sizeof(int));
    //cudaMemset(dhp, static_cast<unsigned char>(H), T * sizeof(int));
    cudaMemcpy(active, &T, sizeof(int), cudaMemcpyHostToDevice);
    //printf("calling kernel\n");

    calculateFinalScore<<<1, T>>>(dx, dy, dscore, dhp, active);
    cudaDeviceSynchronize();
    cudaMemcpy(score, dscore, T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dscore);
    cudaFree(dhp);
    cudaFree(active);
    

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
