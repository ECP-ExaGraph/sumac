#include <stdio.h>


#define gpuErrchk(ans) {gpuAssert((ans), __FILE__,__LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
	if (code != hipSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


void GPU_copy(int numDevices, size_t size, float* elapsed_t){
	

	hipEvent_t start,fin;
	gpuErrchk(hipEventCreate(&start));
	gpuErrchk(hipEventCreate(&fin));

	float** data = new float*[numDevices];
	for(int dev = numDevices-2; dev>=0;dev--){
		gpuErrchk(hipSetDevice(dev));
		hipDeviceEnablePeerAccess(dev+1,0);
		gpuErrchk(hipMalloc(&data[dev],size));

		gpuErrchk(hipSetDevice(dev+1));
		gpuErrchk(hipEventCreate(&start));
		gpuErrchk(hipEventCreate(&fin));
		hipDeviceEnablePeerAccess(dev,0);
		gpuErrchk(hipMalloc(&data[dev+1],size));
		
		gpuErrchk(hipEventRecord(start));
		gpuErrchk(hipMemcpyPeerAsync(data[dev],dev,data[dev+1],dev+1,size));
		gpuErrchk(hipEventRecord(fin));
		gpuErrchk(hipEventSynchronize(fin));
			
		gpuErrchk(hipEventElapsedTime(&elapsed_t[dev],start,fin));		
	
		hipSetDevice(dev);
		hipFree(data[dev]);
		hipSetDevice(dev+1);
		hipFree(data[dev+1]);
	}
	
	hipEventDestroy(start);
	hipEventDestroy(fin);

	free(data);
}



int main(){
	
	int numDevices = 4;
	long dataSize = 512;
	float* elapsed_t = new float[numDevices-1];
	int deviceCount = 0;
	hipGetDeviceCount(&deviceCount);
	printf("NumDevices: %d\n",deviceCount);
	for(dataSize = 512; dataSize<1000000000;dataSize*=2){
	GPU_copy(numDevices,dataSize,elapsed_t);
		for(int t=0;t<numDevices-1;t++){
			//printf("Transfer time (Size = %d) : %d -> %d : %f\n",dataSize, t+1,t,elapsed_t[t]);
			printf("%d,%ld,%f\n",t,dataSize,((dataSize/elapsed_t[t])*1000)/1e9);
		}
		hipDeviceSynchronize();
	}
	free(elapsed_t);	
	return 0;
}
