#include <stdio.h>


#define gpuErrchk(ans) {gpuAssert((ans), __FILE__,__LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


void GPU_copy(int numDevices, size_t size, float* elapsed_t){
	

	cudaEvent_t start,fin;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&fin));

	float** data = new float*[numDevices];
	for(int dev = numDevices-2; dev>=0;dev--){
		gpuErrchk(cudaSetDevice(dev));
		cudaDeviceEnablePeerAccess(dev+1,0);
		gpuErrchk(cudaMalloc(&data[dev],size));

		gpuErrchk(cudaSetDevice(dev+1));
		gpuErrchk(cudaEventCreate(&start));
		gpuErrchk(cudaEventCreate(&fin));
		cudaDeviceEnablePeerAccess(dev,0);
		gpuErrchk(cudaMalloc(&data[dev+1],size));
		
		gpuErrchk(cudaEventRecord(start));
		gpuErrchk(cudaMemcpyPeerAsync(data[dev],dev,data[dev+1],dev+1,size));
		gpuErrchk(cudaEventRecord(fin));
		gpuErrchk(cudaEventSynchronize(fin));
			
		gpuErrchk(cudaEventElapsedTime(&elapsed_t[dev],start,fin));		
	
		cudaSetDevice(dev);
		cudaFree(data[dev]);
		cudaSetDevice(dev+1);
		cudaFree(data[dev+1]);
	}
	
	cudaEventDestroy(start);
	cudaEventDestroy(fin);

	free(data);
}



int main(){
	
	int numDevices = 8;
	long dataSize = 512;
	float* elapsed_t = new float[numDevices-1];
	for(dataSize = 512; dataSize<1000000000;dataSize*=2){
	GPU_copy(numDevices,dataSize,elapsed_t);
		for(int t=0;t<numDevices-1;t++){
			//printf("Transfer time (Size = %d) : %d -> %d : %f\n",dataSize, t+1,t,elapsed_t[t]);
			printf("%d,%ld,%f\n",t,dataSize,((dataSize/elapsed_t[t])*1000)/1e9);
		}
		cudaDeviceSynchronize();
	}
	free(elapsed_t);	
	return 0;
}
