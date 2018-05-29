// Compile: nvcc Roofs.cu -std=c++11 -lcuda -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_imgcodecs -o Roofs 
#include <string> 
#include <fstream> 
#include <iostream> 

#include <opencv2/cudev.hpp> 
#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <cuda.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 

__device__ uchar3 BBGR   = {208,208,208};    // Building colour. 
__device__ uchar3 Orange = {80,127,255};     // Solar orange. 


__global__ void Roofs(int N, uchar3 * DPicture, uchar3 * DOut, int COLS, int ROWS, int ZONE, int * DPixels) 
{
    int PId  = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if (PId >= N) return; 
    
    int Sum = 0, NS = 0; 
    for (int i=-ZONE; i<=ZONE; i++) 
    {
        if (DPicture[PId+i].x  == BBGR.x && DPicture[PId+i].y  == BBGR.y && DPicture[PId+i].z  == BBGR.z) 
            Sum++; 
        NS = COLS*i; 
        if (DPicture[PId+NS].x == BBGR.x && DPicture[PId+NS].y == BBGR.y && DPicture[PId+NS].z == BBGR.z) 
            Sum++; 
    }
    if (Sum == 2*(2*ZONE+1)) 
    {
        DOut[PId] = Orange; 
        atomicAdd(&DPixels[0], 1); 
    }
}


int main(int argc, char ** argv) 
{
    cudaSetDevice(1);   // Use 2nd GTX 1080Ti. 
    std::cout << "CUDA  Status : " << cv::cuda::getCudaEnabledDeviceCount() << " GPUs.\n\n"; 
    
    if (argc < 3) return 0; 
    
    std::ifstream Files(argv[1]); 
    int ZONE    = std::stoi(argv[2]); 
    std::cout  << "File: " << argv[1] << " , Buffer zone = " << ZONE << " pixels. \n"; 
    
    cv::Mat OSM  = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR); 
    cv::Mat Img; 
    cv::cvtColor(OSM, Img, cv::COLOR_BGR2RGB); 
    
    thrust::device_vector<int>      DPix(1); 
    thrust::device_vector<uchar3>   DPicture; 
    thrust::device_vector<uchar3>   DOutput(Img.cols * Img.rows); 
    std::vector<uchar3>             Output(Img.cols  * Img.rows); 
    
    if (!Img.data) return 0; 
    
    if (Img.isContinuous()) 
    { 
        DPicture.assign((uchar3*)Img.datastart, (uchar3*)Img.dataend); 
        DOutput = DPicture; 
        uchar3  * DArray  = thrust::raw_pointer_cast(&DPicture[0]); 
        uchar3  * DOut    = thrust::raw_pointer_cast(&DOutput[0]); 
        int     * DPxl    = thrust::raw_pointer_cast(&DPix[0]); 
        
        Roofs<<<(Img.rows*4)+1, (Img.cols/4)+1>>>(DPicture.size(), DArray, DOut, Img.cols, Img.rows, ZONE, DPxl); 
        
        thrust::copy(DOutput.begin(), DOutput.end(), Output.begin()); 
        cv::Mat Dest(Img.rows, Img.cols, CV_8UC3, (void*)&Output[0]); 
        cv::imwrite("Output.jpg", Dest); 
        std::cout << "Total solar area available = " << DPix[0] << " pixels. \n"; 
    } 
    return 0; 
}


