#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
using namespace std;

/*
 * 	dim3 threadsPerBlock(32, 32);
 *	dim3 blockPerGrid((imgWidth  + threadsPerBlock.x - 1)/threadsPerBlock.x ,
 *	                  (imgHeight + threadsPerBlock.y - 1)/threadsPerBlock.y);
 *
 *	gridDim_x = (imgWidth  + threadsPerBlock.x - 1)/threadsPerBlock.x
 *	gridDim_y = (imgHeight + threadsPerBlock.y - 1)/threadsPerBlock.y
 *

dim3 |  二维grid(gridDim_x, gridDim_y)   |     二维block(32，32)   |
     |     gridDim.x=gridDim_x          |    blockDim.x=32       |
     |     gridDim.y=gridDim_y          |    blockDim.y=32       |
     |     gridDim.z=1                  |    blockDim.z=1        |
     |                                  |                        |
     |     blockIdx.x=[0,gridDim_x-1]   |    threadIdx.x=[0,31]  |
     |     blockIdx.y=[0,gridDim_x-1]   |    threadIdx.y=[0,31]  |
     |     blockIdx.z=0                 |    threadIdx.z=0       |
*/

__global__ void rgb2grayInCuda(uchar3 *dataIn, unsigned char *dataOut,int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x; //二维线程模型的全部x轴线程索引
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y; //二维线程模型的全部y轴线程索引

    if(xIndex < imgWidth && yIndex < imgHeight){
    	uchar3 rgb = dataIn[xIndex + yIndex * imgWidth]; //采用线程一维索引读取数据指针数组
    	dataOut[yIndex * imgWidth + xIndex] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

__global__ void imHistInCuda(unsigned char *dataIn, int *hist)
{
	int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;
	int blockIndex = blockIdx.x + blockIdx.y * gridDim.x;
	int Index = blockIndex * blockDim.x * blockDim.y + threadIndex; //二维线程模型全部一维的线程索引

   atomicAdd(&hist[dataIn[Index]],1); //原子加法操作
}

int main(void)
{
    cv::Mat srcImg = cv::imread("test.jpg");

    int imgWidth = srcImg.cols;
    int imgHeight = srcImg.rows;
    cv::Mat grayImg(imgHeight, imgWidth, CV_8UC1, cv::Scalar(0)); //定义空图像矩阵作为输出灰度图 (h,w,c)
    int hist[256]; //定义灰度直方图像素统计数组
    memset(hist, 0, 256 * sizeof(int)); //初始化数组为0并分配相应大小的内存

    //在GPU中开辟输入输出空间
    uchar3 *d_in; //定义GPU输入图像指针 (三维字符数组类型)
    unsigned char *d_out; //定义GPU输出图像指针
    int *d_hist; //定义GPU输出直方图指针
    cudaMalloc((void**)&d_in, imgWidth * imgHeight * sizeof(uchar3)); //为GPU变量输入图像指针分配内存
    cudaMalloc((void**)&d_out, imgWidth * imgHeight * sizeof(unsigned char)); //为GPU变量输出图像指针分配内存
    cudaMalloc((void**)&d_hist, 256 * sizeof(int)); //为GPU变量输出直方图指针分配内存

    //将图像数据传入GPU中
    cudaMemcpy(d_in, srcImg.data, imgWidth * imgHeight * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, hist, 256 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x -1)/threadsPerBlock.x, (imgHeight + threadsPerBlock.y - 1)/threadsPerBlock.y);

    //灰度化
    rgb2grayInCuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, imgHeight, imgWidth);

    //灰度直方图统计
    imHistInCuda<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_hist);

    //将数据从GPU传回CPU
    cudaMemcpy(grayImg.data, d_out, imgHeight*imgWidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);


    cv::imshow("grayImage", grayImg);
    cv::waitKey();

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);

	return 0;
}
