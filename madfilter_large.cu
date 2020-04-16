/* MAD Filter on GPU 

Version 3.0 
Runs on single bin size
Input:  filename
        Number of samples to filter
        Bin size
        Threshold (multiple of sigma)
        Option for filtering
        Name of timing file

Compile it with following line:
nvcc -Xptxas="-v" -o madfilter_large madfilter_large.cu -arch=sm_20 
And run with
./madfilter_large c3.txt 16384000 16384 3 -z times
./madfilter_large c3.txt 102400 1024 3 -z times
(Rohini Joshi, 2013 - rmjoshi.06@gmail.com)

*/
  
#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include<stdio.h>
#include<sys/time.h>
 
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
	
static void HandleError( cudaError_t err, const char *file, int line ) { 
        if (err != cudaSuccess) {
                cerr<<"**CUDA Error: " << cudaGetErrorString(err)<< "in "<< file <<" at line "<< line << endl; 
                exit( EXIT_FAILURE );
        }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define nbin 2
__device__ __constant__ int total;	//number of bins
__device__ __constant__ int bsize;	//binsize in number of samples
__device__ __constant__ int dop;	//option for to use for filtering

__device__ float randomnumber(int t, int i){

curandState s;
float x;
curand_init(t, 0, i, &s);
x = curand_normal(&s);

return x;
}

__global__ void findrms(int *data, float *RMS){

//Each thread is responsible for finding mean and rms of a block of data of size bsize
int tid = threadIdx.x + blockIdx.x * blockDim.x, lw, up;
int sum, sumsq, sample;
float mean;

//Limits that change for each thread such that it will access a unique block
lw = tid*bsize;
up = lw + bsize;

// if condn is so that data past allocated memory aren't accessed 
if (tid < total){
for (int i=lw; i<up; i++){
	sample = data[i];
	sum += sample;
	sumsq += sample*sample;
}
mean = sum/bsize;
// Store back to global memory
RMS[tid] = sqrtf( sumsq/bsize - mean*mean );
}
}

__global__ void findhist( int *data, unsigned int *hist, int *not_flagged_data, unsigned int *num, int binno, unsigned int *total_effsize){

// Histogram is found in chunks across many blocks and added together
__shared__ unsigned int temphist[256];
int i = threadIdx.x + blockIdx.x * blockDim.x;
int offset = gridDim.x * blockDim.x;
__shared__ unsigned int effsize;

// Initialize shared memory.
// For single locations make sure only one thread does the write to avoid duplication of effort
if (threadIdx.x == 0)
	effsize=0;
if (i==0)
	*total_effsize = 0;

hist[threadIdx.x] = 0;
temphist[threadIdx.x] = 0;
__syncthreads();

// Offset into input array to access current bin
data = data + bsize*(binno);

while (i < bsize){
	// Flagging
        if((data[i]==-128) || (data[i] == 127)){
		i += offset;
        }else{
		// Store data not flagged into separate array
                not_flagged_data[i] = data[i];
		atomicAdd( &effsize, 1 );
		atomicAdd(&temphist[data[i] + 128], 1);
		i += offset;
	}
}

__syncthreads();
atomicAdd( &(hist[threadIdx.x]), temphist[threadIdx.x] );
if (threadIdx.x == 0)
	atomicAdd( total_effsize, effsize );
/*if (*binno==0){
if (threadIdx.x == 0)
	printf("eff %d\n", effsize);
}*/
}

__global__ void findmed( unsigned int *hhist, float *med, int *f1, int *f2, int bins, unsigned int *effsize ){

if ((threadIdx.x + blockIdx.x*blockDim.x) < bins){

int i, c=0, d, flag=0, odd=0, binno = threadIdx.x + blockIdx.x*blockDim.x;
// variable j is effective size of bin
unsigned int j=*(effsize+binno), *hist;
hist = hhist + 256*binno;

/* Find median. Two methods for even/odd sizes. Modify if data is 4 bit
flag = 1/0 if median is floating point/int
odd = 1/0 if data set is odd/even 
median can only be float if data set is even */

if (j%2 == 0){
        d = j/2;
        for ( i=0; i<(256); i++){
                c = c + hist[i];
                if (c==d){
                        *(med+binno) =(float)( (2*(i) + 1)*0.5 - 128 );
                        flag = 1;
                        break;
                }else if (c>d){
                        *(med+binno) = (i - 128);
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
                        *(med+binno) = i - 128;
                        break;
                }
        }
}
*(f1+binno) = flag;
*(f2+binno) = odd;
}
//printf("Median = %f for Bin no %d with size %d\n", *med, *binno, effsize);
}

__global__ void finddev(int *data, float *dev, float *med, unsigned int *effsize){

int i = threadIdx.x + blockIdx.x * blockDim.x;
int offset = gridDim.x * blockDim.x;
float lmed = *med;
while ( i<*effsize ) {
	dev[i] = fabsf( data[i] - lmed );
	i += offset;
}
}

__global__ void findhistdev(float *dev, unsigned int *hist, int *f1, unsigned int *effsize){

__shared__ unsigned int temphist[256];
int i = threadIdx.x + blockIdx.x * blockDim.x, ii;
//int flag = *f1;
int offset = gridDim.x * blockDim.x;

temphist[threadIdx.x] = 0;
__syncthreads();

//if (threadIdx.x == 0)
//	printf("ef %d\n", effsize);
while (i<*effsize){
/*
 if (flag == 0){
	ii = (int)(ceil(dev[i]));
 	atomicAdd(&(temphist[ii]), 1);
 }else{
	int p;
	p = (int) dev[i];
	atomicAdd(&(temphist[p]), 1);

}
*/
	ii = (int)dev[i];
	atomicAdd( &temphist[ii], 1 );
	i += offset;

}
__syncthreads();
atomicAdd( &(hist[threadIdx.x]), temphist[threadIdx.x] );

}

__global__ void findmad(unsigned int *hhist, int *f1, int *f2, float *d_mad, int bins, unsigned int *effsize){

int d, i, s = 0, binno = threadIdx.x + blockIdx.x*blockDim.x;
int flag = *(f1+binno), odd = *(f2+binno);
unsigned int *hist, j;
float mad;
if ((threadIdx.x + blockIdx.x*blockDim.x) < bins){
hist = hhist + 256*binno;
j = *(effsize+binno);
if (flag == 0){
	if (odd == 0){
		d = j/2;
		for (i = 0; i<256; i++){
	                s = s+hist[i];
        	        if (s == d){
                	        mad = (float)( (2*i + 1)*0.5 );
                        	break;
	                }else if (s > d ){
	                        mad = i;
	                        break;
	                }else
	                        continue;
        	}
        }else{
		d = (j + 1)/2;
	        for (i = 0; i<256; i++){
        	        s = s + hist[i];
                	if(s >= d){
	                        mad = i;
        	                break;
                	}
        	}
        }

}else{
	d = j/2;
        for (i = 0; i<256; i++){
                s = s+hist[i];
                if (s == d){
                        mad = (float)( (2*i + 1)*0.5 + 0.5 );
                        break;
                }else if (s > d){
                        mad = (float)( i + 0.5 );
                        break;
                }else
                        continue;
        }

}

d_mad[(binno)]= mad;
}
}

__global__ void filter( int *d_data, float *d_mad, float *d_med, bool *d_flag, int thresh_n ){

//filtering
int i, tid = threadIdx.x + blockIdx.x * blockDim.x;
int lw = tid * (bsize);
int up = lw + (bsize);
float med, mad, thresh;

// int sum=0, sumsq=0;
// float mean;

if (tid < (total)){
thresh = thresh_n*1.4826*d_mad[tid];
mad = d_mad[tid];
med = d_med[tid];

for( i=lw; i<up; i++){
	if ( (abs(d_data[i]) > thresh) || (d_data[i] == -128) || (d_data[i] == 127)  ){
        	if(dop == 0){
	                d_data[i] = 0;
	        }else if(dop == 1){
        	        d_data[i] = med;
		}else if(dop == 2){
			d_data[i] = rint(0 + 1.4826*mad*randomnumber(tid, i-lw));
	        }else if(dop == 3){
			d_data[i] = thresh;
		}d_flag[i] = 0;
	}
	else{
		d_flag[i] = 1;
	}
//	sum += d_data[i];
//	sumsq += d_data[i]*d_data[i];
}

/* Find RMS after filtering */
//mean = sum/(bsize);
//d_rms_a[tid] = sqrtf( sumsq/(bsize) - mean*mean );

}
}

  
int main(int argc, char *argv[]){
int i, n, num, size, binsize, binno, bins;
int *h_data, *d_data, *temp, *med_flag, *odd;
unsigned int *hist, *histdev, *d_num, *effsize;
float *h_rms_b, *h_rms_a, *d_rms_b, *d_rms_a, *h_mad, *d_mad, *med, *dev;
double time1, time2;
FILE *fp;
char *fname, *ffname,*op, *stat_file;
struct timeval tim;
float dtime1, time_initial, time_findmad, time_filter, time_computation, time_copyback, time_total;
bool *h_flag, *d_flag;
cudaDeviceProp prop;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

if (argc < 6){
	system("./help_large.sh");
	exit(0);
}	
fname = argv[1];		// filename
size = atoi( argv[2] );		// Number of elements to take from this file
binsize = atoi( argv[3] );	// Size of one bin (in samples)
n = atoi( argv[4] );		// Multiple of MAD to use as a threshold
op = argv[5];			// Option to use for filtering. What to replace RFI with?
stat_file = argv[6];		// Timings will be written to this file

// Number of bins
bins = (int)size/binsize;
// Total size as a multiple of the bin size
size = bins*binsize;

gettimeofday(&tim, NULL);
time1 = tim.tv_sec + (tim.tv_usec/1000000.0);

/* Allocate and store input on host */
h_data = (int *)malloc(size*sizeof(int));
h_rms_b = (float *)malloc(bins*sizeof(float));
h_rms_a = (float *)malloc(bins*sizeof(float));
h_mad = (float *)malloc(bins*sizeof(float));
h_flag = (bool *)malloc(size*sizeof(bool));

// For debugging purpose only
float *h_temp;
h_temp = (float *)malloc(sizeof(float)*binsize);
int *h_med_flag, *h_odd;
h_med_flag = (int *)malloc(sizeof(int) * bins);
h_odd = (int *)malloc(sizeof(int) * bins);
//New file name ffname to store filtered data
ffname = (char *)malloc(30*sizeof(char));
sprintf(ffname, "%s_filtered", fname);

gettimeofday( &tim, NULL );
double time15 = tim.tv_sec + (tim.tv_usec/1000000.0);

fp = fopen(fname, "r");
if (fp == NULL){
        printf("Error in opening input file\n");
}
for(i=0; i<size; i++){
        fscanf(fp, "%d\n", &num);	// Store in the data (8 bit integers)
        h_data[i] = num;
}
fclose(fp);

// As strcmp cannot be used in a kernel, convert the char option to integer here
if(!strcmp(op, "-z")){
	i=0;
}else if(!strcmp(op, "-m")){
	i=1;
}else if(!strcmp(op, "-r")){
	i=2;
}else if(!strcmp(op, "-c")){
	i=3;
}
gettimeofday(&tim, NULL);
time2 = tim.tv_sec + (tim.tv_usec/1000000.0);
dtime1 = time2 - time1; //dtime1 contains time required for initialization stuff on the host (only) in seconds.
cerr << "Time to allocate memory on the host = "  << time15-time1 <<" sec" << endl;
cerr << "Time to store data on host from file = " << time2-time15 << " sec" << endl;
cerr << "Total time for initialization on host = " << dtime1 << " sec" << endl;
 
/* Allocate i/o and store input on device */
cudaEventRecord( start, 0 );
HANDLE_ERROR( cudaMalloc( (void **)&d_data, size*sizeof(int) ) );
HANDLE_ERROR( cudaMalloc( (void **)&d_rms_b, bins*sizeof(float) ) );
HANDLE_ERROR( cudaMalloc( (void **)&d_rms_a, bins*sizeof(float) ) );
HANDLE_ERROR( cudaMalloc( (void **)&d_mad, bins*sizeof(float) ) );
HANDLE_ERROR( cudaMalloc( (void **)&dev, sizeof(float)*binsize ) );
HANDLE_ERROR( cudaMalloc( (void **)&temp, sizeof(int)*binsize*bins ) );
HANDLE_ERROR( cudaMalloc( (void **)&d_flag, size*sizeof(bool) ) );
HANDLE_ERROR( cudaMalloc( (void **)&hist, 256 * bins * sizeof(unsigned int)) );
HANDLE_ERROR( cudaMalloc( (void **)&histdev, 256 * sizeof(unsigned int)) );
HANDLE_ERROR( cudaMalloc( (void **)&med, sizeof(float)*bins ) );
HANDLE_ERROR( cudaMalloc( (void **)&d_num, sizeof(unsigned int) ) );
HANDLE_ERROR( cudaMalloc( (void **)&med_flag, sizeof(int)*bins ) );
HANDLE_ERROR( cudaMalloc( (void **)&odd, sizeof(int)*bins ) );
HANDLE_ERROR( cudaMalloc( (void **)&effsize, sizeof(unsigned int)*bins ) );


HANDLE_ERROR( cudaMemcpy( d_data, h_data, size*sizeof(int), cudaMemcpyHostToDevice ) );
HANDLE_ERROR( cudaMemcpyToSymbol( "bsize", &binsize, sizeof(int) ) );
HANDLE_ERROR( cudaMemcpyToSymbol( "total", &bins, sizeof(int) ) );
HANDLE_ERROR( cudaMemcpyToSymbol( "dop", &i, sizeof(int) ) );
HANDLE_ERROR( cudaMemset( temp, 0, binsize * bins * sizeof(int) ) );
HANDLE_ERROR( cudaMemset( hist, 0, 256 * bins * sizeof(unsigned int) ) );

cudaEventRecord( stop, 0 );
cudaEventElapsedTime( &time_initial, start, stop);
// dtime2 is time required for allocation and mem copy/set on the device in milliseconds
cerr << "Time for allocation of memory and copying host->device = " << time_initial << " msec" << endl;
/*printf("Memory allocated and set on device.\n");
printf("Time required for host = %f sec\nTime required for device = %f sec\nTotal time = %f sec\n", dtime1, dtime2/1000.0, dtime2/1000.0 + dtime1);
// dtime1 now contains total doing-initial-jazz time in sec
dtime1 += dtime2/1000.0;
*/


int blocks = 1, threads = 256;
if (bins>threads )
	blocks = ceil(bins/(float)threads);
else
	threads = bins;
findrms<<<blocks, threads>>>(d_data, d_rms_b);

printf("Number of bins = %d\nfindrms\nGrid dim [%d 1] Block dim [%d 1]\n", bins, blocks, threads);

cudaGetDeviceProperties( &prop, 0 );
printf("%d\n", prop.multiProcessorCount );
cudaEventRecord(start, 0);

// find hist serially over bins
for (binno = 0; binno < bins; binno++){

//	HANDLE_ERROR( cudaMemcpy( d_binno, &binno, sizeof(int), cudaMemcpyHostToDevice ) );
//	HANDLE_ERROR( cudaMemset( hist, 0, 256 * sizeof(unsigned int) ) );
//	cudaDeviceSynchronize();
	//HANDLE_ERROR( cudaMemset( &effsize, 0, sizeof(unsigned int) ) );

        // 256 is number of data given to a block for hist
        // hist is series of histograms for each bin
	threads = 256; blocks=binsize/threads;   // blocks is nblocks
 	findhist<<<blocks, threads>>>(d_data, hist+256*binno, temp + binsize*binno, d_num, binno, effsize+binno);

//	cudaDeviceSynchronize();
	//cudaError_t error = cudaGetLastError();
	//printf("%s\n", cudaGetErrorString(error) );
	/*if (binno == 1){
		unsigned int h_hist[256];
		FILE *fp;
		HANDLE_ERROR( cudaMemcpy( h_hist, hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		fp = fopen("hist.dat", "w");
		for (int i=0; i<256; i++)
			fprintf(fp, "%d\n", h_hist[i]);
		fclose(fp);
	}*/
}

//	printf("1\n");
//for (binno = 0; binno < bins; binno++){

        // parallelise median finding over bins now
	threads = 256; blocks = (bins+threads-1)/threads;  
	findmed<<<blocks,threads>>>(hist, med, med_flag, odd, bins, effsize);

	//error = cudaGetLastError();
	//printf("%s\n", cudaGetErrorString(error) );
	//printf("2\n");
//}	

// serialise y-median over bins
for (binno = 0; binno < bins; binno++){
	threads=256; blocks = binsize/(2*threads);
	finddev<<<blocks, threads>>>(temp + binsize*binno, dev, med + binno, effsize+binno); 

	//error = cudaGetLastError();
	//printf("%s\n", cudaGetErrorString(error) );
	//printf("3\n");
	
	
	/*if (binno == 0){
		FILE *fp;
		HANDLE_ERROR( cudaMemcpy( h_temp, temp, binsize*sizeof(float), cudaMemcpyDeviceToHost ) );
		fp = fopen("dev.dat", "w");
		for (int i=0; i<binsize; i++)
			fprintf(fp, "%d\n", h_temp[i]);
		fclose(fp);
	}*/
//	HANDLE_ERROR( cudaMemset( hist, 0, 256 * sizeof(unsigned int) ) );

	threads=256;blocks= binsize/threads;
	findhistdev<<<blocks, threads>>>(dev, hist+256*binno, med_flag + binno, effsize+binno);

//	cudaThreadSynchronize();
	//error = cudaGetLastError();
	//printf("%s\n", cudaGetErrorString(error) );
	//printf("4\n");
}

//for (binno = 0; binno < bins; binno++){

        // parallelse over bins to find median of dev
	threads = 256; blocks = (bins+threads-1)/threads; 
	findmad<<<blocks, threads>>>(hist, med_flag, odd, d_mad, bins, effsize);
//	cudaThreadSynchronize();
	//error = cudaGetLastError();
	//printf("%s\n", cudaGetErrorString(error) );
	//printf("5\n");
//}

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime( &time_findmad, start, stop);
cudaEventRecord(start, 0);

blocks = 1, threads = 32;
if (bins>threads)
	blocks = ceil(bins/(float)threads);
else
	threads = bins;

printf("Filter\nGrid dim [%d 1] Block dim [%d 1]\n", blocks, threads);
filter<<<blocks, threads>>>( d_data, d_mad, med, d_flag, n );

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime( &time_filter, start, stop);
time_computation = time_findmad + time_filter;

blocks = 1, threads = 256;
if (bins>threads )
	blocks = ceil(bins/(float)threads);
else
	threads = bins;
findrms<<<blocks,threads>>>( d_data, d_rms_a );

//gettimeofday(&tim, NULL);
//double timex = tim.tv_sec * 1000.0 + (tim.tv_usec/1000.0);
cudaEventRecord(start, 0);

/* Copy data back to host */

HANDLE_ERROR( cudaMemcpy( h_flag, d_flag, size*sizeof(bool), cudaMemcpyDeviceToHost ) );
//HANDLE_ERROR( cudaDeviceSynchronize() );
//gettimeofday(&tim, NULL);
//double timey = tim.tv_sec * 1000.0 + (tim.tv_usec/1000.0);
//time_copyback = timey - timex;
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime( &time_copyback, start, stop);

HANDLE_ERROR( cudaMemcpy( h_mad, d_mad, bins*sizeof(float), cudaMemcpyDeviceToHost ) );
HANDLE_ERROR( cudaMemcpy( h_data, d_data, size*sizeof(int), cudaMemcpyDeviceToHost ) );
HANDLE_ERROR( cudaMemcpy( h_rms_b, d_rms_b, bins*sizeof(float), cudaMemcpyDeviceToHost ) );
HANDLE_ERROR( cudaMemcpy( h_rms_a, d_rms_a, bins*sizeof(float), cudaMemcpyDeviceToHost ) );
HANDLE_ERROR( cudaMemcpy( h_med_flag, med_flag, bins*sizeof(int), cudaMemcpyDeviceToHost ) );
HANDLE_ERROR( cudaMemcpy( h_odd, odd, bins*sizeof(int), cudaMemcpyDeviceToHost ) );
HANDLE_ERROR( cudaFree(d_data) );
HANDLE_ERROR( cudaFree(d_mad) );
HANDLE_ERROR( cudaFree(d_rms_b) );
HANDLE_ERROR( cudaFree(d_rms_a) );
HANDLE_ERROR( cudaFree(dev) );
HANDLE_ERROR( cudaFree(temp) );
HANDLE_ERROR( cudaFree(d_flag) );
HANDLE_ERROR( cudaFree(med_flag) );
HANDLE_ERROR( cudaFree(effsize) );

cudaEventDestroy(start); 
cudaEventDestroy(stop); 

time_total = time_initial + time_computation + time_copyback;
cerr << "Time for finding MAD = " << time_findmad << " ms" << endl;
cerr << "Time for filtering data = " << time_filter << " ms" << endl;
cerr << "Total time for computation = " << time_computation << " ms" << endl;
cerr << "Time for copying back to host = " << time_copyback << " ms" << endl;
cerr << "Total time = " << time_total << " ms" << endl;
// 10 ns sampling of data (in c3) 8 bit data
float realtime = 0.000010 * binsize * bins;
cerr << "Data is of " << realtime << " ms" << endl;
if (realtime > (time_total)) 
	cerr << "In real time! By Factor of " << (time_total)/realtime << endl;
else
	cerr << "Not in real time by factor of " << (time_total)/realtime << endl;

freopen (stat_file,"a",stdout);
printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", size, binsize, time_initial, time_findmad, time_filter,time_computation, time_copyback, time_total, realtime );
fclose(stdout);

fp = fopen("mad.dat", "w");
if (fp == NULL){
        printf("Error in opening output file\n");
}
for(i=0; i<bins; i++){
        fprintf(fp, "%f\t%f\t%f\n", h_rms_b[i], h_rms_a[i], h_mad[i]);
}
fclose(fp);
fp = fopen(ffname, "w");
if (fp == NULL){
        printf("Error in opening output file\n");
}
for(i=0;i<size;i++){
	fprintf(fp, "%d\t%d\n", h_data[i], h_flag[i]);
}
fclose(fp);
fp = fopen("flags.dat", "w");
if (fp == NULL){
        printf("Error in opening output file\n");
}
for(i=0; i<bins; i++){
	fprintf(fp, "%d\t%d\n", h_med_flag[i], h_odd[i]);
}
fclose(fp);
printf("Data copied back to host\n");
}
