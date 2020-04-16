/* MAD Filter on GPU

Version 1.0
Runs on single bin size
Input:  filename
        Number of samples to filter
        Bin size
        Threshold (multiple of sigma)
        Option for filtering
	Header size in bytes
 
Basic version using histogram method for median. 24/01/13
Replace with random numbers. Added 28/01/13
Finding mean and rms before and after filtering. Added 02/02/13 
Copy back only flags file (bool)

Compile it with following line:
nvcc -Xptxas="-v" -o madfilter_small madfilter_small.cu -arch=sm_20 

(Rohini Joshi, 2013 - rmjoshi.06@gmail.com)

*/

/* Modified the code to work with 8-bit unsigned data 

 Kaustubh Rajwade (Sept 2017 - Manchester)*/
  
#include<cuda.h>
#include<curand.h>  		// random num generation in cuda
#include<curand_kernel.h>       // random num generation in cuda
#include<stdio.h>
#include<sys/time.h>
#include<iostream>
#include<math.h>
#include<string.h>


using std::cerr;
using std::cout;
using std::endl;

// Setting up device
bool SetGPU()
{
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    char desiredDeviceName[1024];
    strcpy(desiredDeviceName,"GeForce GTX 1080 Ti");
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if (deviceProperties.name == desiredDeviceName)
        {
            cudaSetDevice(1);
            return true;
        }
    }

    return false;
}
 
// __device__ --> is a dev fn to be run on GRID and can be called only from kernel or device fn

__device__ float randomnumber(int t, int i){

curandState s; 
float x;
// curand_init() sets up an initial state s. with seed t(thread id) and sequence number 0 and offset i
// Each bin is filtered with a separate thread. Thus normal distribution of random numbers is preserved within a bin
curand_init(t, 0, i, &s);  // t is a seed, i is offset in seq of random numbers 
// Generate random number from normal distribution
x = curand_normal(&s);

return x;
}

__global__ void madfilter( int *d_data, int binsize, int bins, int op, float *dev, int *not_flagged_data, bool *d_flag, float *d_rms_b, float *d_rms_a, float *d_mad, int mult, time_t currTime){
    
// {0} initialised the whole array. blockDim = number of threads/block=32, tid indexes all threads in the grid
// everything below runs for each thread through threadIdx.x and blockIdx.x
int i, j=0, c=0,d,flag=0,odd=0,sum=0, sumsq=0,histdev[256] = {0},hist[256] = {0}, tid = threadIdx.x + blockIdx.x * blockDim.x;
int lw = tid * (binsize);   // the index in the original data array for each bin beginning
int up = lw + (binsize);    //                    same                           end
float mean, med, mad, thresh;

// variable j is to store effective size of bin (after flagging extremities)
if (tid < bins){    // end crap is not accessed

/* Flagging and generating histogram */
for ( i=lw; i<up; i++){
	sum += d_data[i];
	sumsq += d_data[i]*d_data[i];

	// Flag extremities
        if((d_data[i]==0) || (d_data[i] == 255)){
                continue;
        }else{
                hist[d_data[i]] += 1;
                not_flagged_data[lw+j] = d_data[i];
		j+=1;
        }
}
/* Find RMS before filtering */
mean = sum/(binsize);
d_rms_b[tid] = sqrtf( sumsq/(binsize) - mean*mean );
sum = 0;sumsq = 0;
/* Find median. Two methods for even/odd sizes. Modify if data is 4 bit
flag = 1/0 if median is floating point/int
odd = 1/0 if data set is odd/even 
median can only be float if data set is even */

if (j%2 == 0){
        d = j/2;
        for ( i=0; i<(256); i++){
                c = c + hist[i];
                if (c==d){
                        med =(float)( (2*(i) + 1)*0.5 );
                        flag = 1;
                        break;
                }else if (c>d){
                        med = i;
                        break;
                }else
                        continue;
        }
}else{
        d = (j + 1)/2;
        odd = 1;
        c = 0;
        for ( i=0; i<(256); i++){
                c = c + hist[i];
                if (c >= d){
                        med = i;
                        break;
                }
        }
}
//  MAD
int s = 0, ii;
if (flag == 0){
        for ( i=lw; i<lw+j; i++){
                dev[i] = fabs( not_flagged_data[i] - med );
                ii = (int)(ceil(dev[i]));
		histdev[ii] += 1;
        }
        /* two submethods for even/odd data sets */
        if (odd == 0){
        for ( i=0; i<(256); i++){
                s = s+histdev[i];
                if (s == d){
                        mad = (float)( (2*(i) + 1)*0.5 );
                        break;
                }else if (s > d ){
                        mad = i;
                        break;
                }else
                        continue;
        }
        }else{
        for ( i=0; i<(256); i++){
                s = s + histdev[i];
                if(s >= d){
                        mad = i;
                        break;
                }
        }
        }
}else{
        int p;
        for ( i=lw; i<lw+j; i++){
                dev[i] = (float)fabs( not_flagged_data[i] - med );
                p = (int) dev[i];
                histdev[p] += 1;
        }
        int s = 0;
        d = j/2;
        for ( i=0; i<(256); i++){
                s = s+histdev[i];
                if (s == d){
                        mad = (float)( (2*(i) + 1)*0.5 + 0.5 );
                        break;
                }else if (s > d){
                        mad = (float)( i + 0.5 );
                        break;
                }else
                        continue;
        }

}
  
d_mad[tid] = mad;
thresh = mult*1.4826*mad;
//filtering

// thresh = mult*1.48*mad
// if abs(d-med) > thresh ---> flag

for( i=lw; i<up; i++){
	if ( (fabsf(d_data[i]-med) > thresh) || (d_data[i] == 0) || (d_data[i] == 255)  ){
        	if(op == 0){
	                d_data[i] = 0;
	        }else if(op == 1){
        	        d_data[i] = med;
		}else if(op == 2){
			d_data[i] = rint(mean + 1.4826*mad*randomnumber(currTime, i-lw));
	        }else if(op == 1){
			d_data[i] = thresh;
		}d_flag[i] = 0;
	}
	else{
		d_flag[i] = 1;
	}
	sum += d_data[i];
	sumsq += d_data[i]*d_data[i];
}

/* Find RMS after filtering */
mean = sum/(binsize);
d_rms_a[tid] = sqrtf( sumsq/(binsize) - mean*mean );

/*if(*op == 0){
	printf("replace with zero\n");
}else if(*op == 1){ 
	printf("replace with median\n");
}else if(*op == 2){    
         printf("replace with random number\n");
}
*/

}
}
  
int main(int argc, char *argv[]){

int i, k, mult_thresh, size, bsize, bins, headersize;
int *h_data, *d_data, *not_flagged_data, op_int, num,ex;
float *h_rms_b, *h_rms_a, *d_rms_b, *d_rms_a, *h_mad, *d_mad, *dev;
double time1, time2;
FILE *fp;
char *fname, *ffname,*op;
struct timeval tim;
float time_initial_host, time_initial_dev, time_kernel, time_copyback;
bool *h_flag, *d_flag;

//Define and create CUDA events start and stop for timing GPU activity
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Parse input arguments
// filename	
fname = argv[1];
// Size of data to filter from the input file
size = atoi( argv[2] );
// Bin size to use
bsize = atoi( argv[3] );
// Multiple of MAD to use as threshold 
mult_thresh = atoi( argv[4] );
// Option to use for filtering (what to replace RFI with)
op = argv[5];
// Header size of filterbank file
headersize = atoi(argv[6]);

if (argc <= 5 ){
	system("./help_small.sh");
	exit(0);
}

SetGPU();

// Number of whole bins that can be filtered in the dataset
bins = (int)size/bsize;
// size is now made a multiple of the bin size
size = bins*bsize;

gettimeofday(&tim, NULL);
time1 = tim.tv_sec + (tim.tv_usec/1000000.0);

/* Allocate and store input on host */
h_data = (int *)malloc(size*sizeof(int));	// actual data - will be read from SHM
h_rms_b = (float *)malloc(bins*sizeof(float));	// RMS before filtering for each bin    - for checking
h_rms_a = (float *)malloc(bins*sizeof(float));	// RMS after filtering for each bin    - for checking
h_mad = (float *)malloc(bins*sizeof(float));	// MAD value for each bin    - for checking
h_flag = (bool *)malloc(size*sizeof(bool));	// Flags 
ffname = (char *)malloc(256*sizeof(char));	// New file name in which filtered data will be written out - will become SHM
sprintf(ffname, "%s_filtered", fname);

// Store data in host memory from file   --- will change to reading from SHM continuously
fp = fopen(fname, "r");
if (fp == NULL){
        printf("Error in opening input file\n");
}

// Skipping the header
fseek(fp,headersize,SEEK_SET);

// Reading data as 4 byte integers
for(i = 0; i < size/4; i++) {
	fread((void*) &num,sizeof(int), 1, fp);
        // Bit shift operation to parse 8 bit numbers
        for (k = 0; k < 4; k++) {
          ex = pow(256, k);
          h_data[4*i+k] = (num & (255 * ex)) >> 8 * k;
        }
}

fclose(fp);

// As strcmp cannot be used in a kernel, convert the filtering option from char to integer here itself
if(!strcmp(op, "-z")){
	op_int=0;
}else if(!strcmp(op, "-m")){
	op_int=1;
}else if(!strcmp(op, "-r")){
	op_int=2;
}else if(!strcmp(op, "-c")){
	op_int=3;
}
gettimeofday(&tim, NULL);
time2 = tim.tv_sec + (tim.tv_usec/1000000.0);
time_initial_host = time2 - time1;

/* Allocate i/o and store input on device */
cudaEventRecord( start, 0 ); // Start CUDA timer
cudaMalloc( (void **)&d_data, size*sizeof(int) );
cudaMalloc( (void **)&d_rms_b, bins*sizeof(float) );    // dont need this
cudaMalloc( (void **)&d_rms_a, bins*sizeof(float) );    // dont need this
cudaMalloc( (void **)&d_mad, bins*sizeof(float) );
cudaMalloc( (void **)&dev, bins*sizeof(float)*bsize );
cudaMalloc( (void **)&not_flagged_data, bins*sizeof(int)*bsize );
cudaMalloc( (void **)&d_flag, size*sizeof(bool) );	// actual flags, can be bool
cudaMemcpy( d_data, h_data, size*sizeof(int), cudaMemcpyHostToDevice );

cudaEventRecord( stop, 0 );	// Stop and store time elapsed
cudaEventSynchronize(stop);
cudaEventElapsedTime( &time_initial_dev, start, stop);

/* Setup grid and run kernel */
int blocks, threads = 32;
blocks = (bins + threads - 1)/threads;	// Mathematically equivalent to a ceil(bins/threads) = number of blocks so that 1 thread/bin
printf("Grid dim [%d 1] Block dim [%d 1]\n", blocks, threads);

cudaEventRecord(start, 0);   // to start timing

// Seeding using current time

time_t currTime = time(NULL);

// send # of blocks and threads to the cuda kernel. dev is y-median(y), d_flag is array of bools, mult_thresh=3
// is asynchronous => comes back to cpu even before finishing
madfilter<<<blocks, threads>>>( d_data, bsize, bins, op_int, dev, not_flagged_data, d_flag, d_rms_b, d_rms_a, d_mad, mult_thresh, currTime);

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);   // makes sure gpu is done, and is part of the timing module. but can synchronise in other ways instead 
			      //if not insterested in timing
cudaEventElapsedTime( &time_kernel, start, stop);

printf("Number of Bins = %d\n", bins);
printf("Time for executing kernel = %f msec\n", time_kernel);

/* Copy data back to host */
cudaEventRecord(start, 0);

cudaMemcpy( h_flag, d_flag, size*sizeof(bool), cudaMemcpyDeviceToHost );
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime( &time_copyback, start, stop);
cudaMemcpy( h_mad, d_mad, bins*sizeof(float), cudaMemcpyDeviceToHost );
cudaMemcpy( h_data, d_data, size*sizeof(int), cudaMemcpyDeviceToHost );
cudaMemcpy( h_rms_b, d_rms_b, bins*sizeof(float), cudaMemcpyDeviceToHost );
cudaMemcpy( h_rms_a, d_rms_a, bins*sizeof(float), cudaMemcpyDeviceToHost );

// Free memory on the device
cudaFree(d_data);
cudaFree(d_mad);
cudaFree(d_rms_b);
cudaFree(d_rms_a);
cudaFree(dev);
cudaFree(not_flagged_data);
cudaFree(d_flag);
cudaEventDestroy(start); 
cudaEventDestroy(stop); 
cerr << "Time to copyback = " << time_copyback << " ms" << endl;
cerr << "Total time = " << time_copyback + time_kernel << " ms" << endl;

// Write out to file
fp = fopen("mad.dat", "w");
if (fp == NULL){
        printf("Error in opening output file\n");
}
for(i=0; i<bins; i++){
        fprintf(fp, "%f\t%f\t%f\n", h_rms_b[i], h_rms_a[i], h_mad[i]);
}
fclose(fp);
fp = fopen(ffname, "wb");
if (fp == NULL){
        printf("Error in opening output file\n");
}
// Changing int to unsigned char

unsigned char* h_data_u;

h_data_u=(unsigned char *)malloc(sizeof(unsigned char)*size);

for (i=0;i<size;i++){
  h_data_u[i] = (h_data[i] & (255));
}

// Writing out the filtered file
fwrite(h_data_u, sizeof(unsigned char), size, fp);

// Free all arrays on the host
free(h_data_u);
free(h_data);
free(h_rms_a);
free(h_rms_b);
free(h_flag);
free(h_mad);
free(ffname);

//Closing file
fclose(fp);
printf("Data copied back to host\n");
return(0);

}
