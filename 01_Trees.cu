// Compile: nvcc 01_Trees.cu -std=c++11 -lcuda -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_imgcodecs -o Trees 
#include <string> 
#include <fstream> 
#include <iostream> 

#include <opencv2/cudev.hpp> 
#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <cuda.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 

const int GMIN = 20; 
const int GMAX = 64; 
const int CELL = 80; 
const int CMAX = 288; 
const int AREA = CELL*CELL; 


__global__ void Trees(int N, uchar3 * DPicture, int * DCells, int COLS, int CSize, int CCOLS) 
{
    int PId = (blockIdx.x * blockDim.x) + threadIdx.x; 
    int RX  = (PId / COLS)  / CSize; 
    int CX  = (PId % COLS)  / CSize; 
    int CId = (RX  * CCOLS) + CX; 
    if (PId >= N  || CId >= CMAX) return; 
    
    atomicAdd(&DCells[CId], DPicture[PId].y); 
}

struct CSV 
{
    std::string Name; 
    int   H;    int S; int V; 
    float Lat;  float  Lon;  float Scale; 
}; 


int main(int argc, char ** argv) 
{
    cudaSetDevice(1);   // Use 2nd GTX 1080Ti. 
    std::cout << "CUDA  Status : " << cv::cuda::getCudaEnabledDeviceCount() << " GPUs.\n\n"; 
    
    std::vector<CSV>    Inputs; 
    std::ifstream       Files(argv[1]); 
    CSV                 This = {"", 0, 0, 0, 0.0f, 0.0f, 0.0f}; 
    
    while (Files >> This.Name >> This.H >> This.S >> This.V >> This.Lat >> This.Lon >> This.Scale) 
    {
        Inputs.push_back(This); 
    }
    for (int A=0; A<Inputs.size(); A++) 
    {
        This = Inputs[A]; 
        std::cout << "File  " << This.Name << ", " << This.H << " " << This.S << " " << This.V << "; "; 
        std::cout << This.Lat << ", " << This.Lon  << "; "   << This.Scale << "\n"; 
        
        cv::Mat Img  = cv::imread(This.Name, CV_LOAD_IMAGE_COLOR); 
        cv::Mat Hsv, Mask, Res; 
        
        thrust::host_vector<uchar3>     Picture; 
        thrust::device_vector<uchar3>   DPicture; 
        thrust::device_vector<int>      DGreens(Img.rows * Img.cols / AREA); 
        
        if (!Img.data) 
        { 
            std::cout << "Couldn't read " << This.Name << " .\n"; 
            continue; 
        } 
        if (Img.isContinuous()) 
        { 
            cv::cvtColor(Img,    Hsv,       cv::COLOR_BGR2HSV); 
            cv::inRange(Hsv,     cv::Scalar(This.H-40,This.S-90,This.V-50), cv::Scalar(This.H+40,This.S+90,This.V+50), Mask); 
            cv::bitwise_and(Img, Img,       Res, Mask); 
            cv::cvtColor(Res,    Img,       cv::COLOR_BGR2RGB); 
            
            Picture.assign((uchar3*)Img.datastart, (uchar3*)Img.dataend); 
            DPicture = Picture; 
            uchar3   * DArray = thrust::raw_pointer_cast(&DPicture[0]); 
            int      * DCells = thrust::raw_pointer_cast(&DGreens[0]); 
            
            Trees<<<(Img.rows*2)+1, (Img.cols/2)+1>>>(DPicture.size(), DArray, DCells, Img.cols, CELL, Img.cols/CELL); 
            
            std::string   Fname = argv[2] + std::to_string(A+1); 
            std::ofstream File(Fname + ".txt"); 
            for (int T=0; T<CMAX; T++) 
            {
                if (DGreens[T] > AREA*GMIN && DGreens[T] < AREA*GMAX) 
                    File << "T"; 
                else 
                    File << "-"; 
                if ((T+1)%(Img.cols/CELL) == 0) File << "\n"; 
            }
            imwrite(Fname + ".jpg", Img); 
        } 
    }
    return 0; 
}

