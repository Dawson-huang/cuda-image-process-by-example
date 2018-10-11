#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>

#include <opencv2/opencv.hpp>
#include <iostream>
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

//腐蚀
__global__ void erodeInCude(unsigned char *dataIn, unsigned char *dataOut, cv::Size erodeElement, int imgWidth, int imgHeight)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x; //二维线程模型的全部x轴线程索引(二维索引写法)
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y; //二维线程模型的全部y轴线程索引

    int Index = xIndex + yIndex * imgWidth;//（一维索引写法）

    int elementWidth = erodeElement.width;
    int elementHeight = erodeElement.height;
    int halfEW = elementWidth / 2;
    int halfEH = elementHeight / 2;

    //初始化输出图
    dataOut[Index] = dataIn[Index];

    //防止越界  halfEW < xIndex < imgWidth-halfEW         halfEH < yIndex < imgHeight-halfEH
    if (xIndex > halfEW && xIndex < imgWidth-halfEW && yIndex > halfEH && yIndex < imgHeight-halfEH)
    {
    	for (int i = -halfEH; i < halfEH + 1; i++)
    	{
    		for (int j = -halfEW; j < halfEW + 1; j++)
    		{
                        //腐蚀：像素比大小
    			if (dataIn[(i + yIndex) * imgWidth + xIndex + j] < dataOut[yIndex * imgWidth + xIndex])
    			{
    				dataOut[yIndex * imgWidth + xIndex] = dataIn[(i + yIndex) * imgWidth + xIndex + j];
    			}
    		}
    	}

    }

}

//膨胀
__global__ void dilateInCuda(unsigned char *dataIn, unsigned char *dataOut, cv::Size dilateElement, int imgWidth, int imgHeight)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.y; //二维线程模型中Grid的x轴索引线程索引(二维线程写法)
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y; //二维线程模型中Grid的y轴索引线程索引

    int elementWidth = dilateElement.width;   // 算子宽
    int elementHeight = dilateElement.height; // 算子高
    int halfEW = elementWidth / 2;
    int halfEH = elementHeight / 2;

    //初始化输出图
    dataOut[yIndex * imgWidth + xIndex] = dataIn[yIndex * imgWidth + xIndex];

    //防止越界
    if (xIndex > halfEW && xIndex < imgWidth - halfEW && yIndex > halfEH && yIndex < imgHeight - halfEH)
    {
    	for(int i = -halfEH; i < halfEH + 1; i++)
    	{
    		for(int j = -halfEW; j < halfEW + 1; j++)
    		{
    			if (dataIn[(i + yIndex) * imgWidth + xIndex + j] > dataOut[yIndex * imgWidth + xIndex])
    			{
    				dataOut[yIndex * imgWidth + xIndex] = dataIn[(i + yIndex) * imgWidth + xIndex + j];
    			}
    		}
    	}
    }

}


int main()
{
	cv::Mat srcImg = cv::imread("/home/eryuan/eryuan_project/cuda-workspace/cuda_image_process_erode_dilate/src/test.jpg");
	cv::Mat grayImg = cv::imread("/home/eryuan/eryuan_project/cuda-workspace/cuda_image_process_erode_dilate/src/test.jpg", 0); //读取灰度图

    unsigned char *d_in;   //输入GPU的图片指针变量
    unsigned char *d_out1; //GPU输出的腐蚀图片指针变量
    unsigned char *d_out2; //GPU输出的膨胀图片指针变量

    int imgWidth = grayImg.cols;
    int imgHeight = grayImg.rows;

    cv::Mat erodeImg(imgHeight, imgWidth, CV_8UC1, cv::Scalar(0));  //定义空图像存放腐蚀结果
    cv::Mat dilateImg(imgHeight, imgWidth, CV_8UC1, cv::Scalar(0)); //定义空图像存放膨胀结果

    //为GPU变量指针分配GPU内存
    cudaMalloc((void**)&d_in, imgWidth * imgHeight * sizeof(unsigned char));
    cudaMalloc((void**)&d_out1, imgWidth * imgHeight * sizeof(unsigned char));
    cudaMalloc((void**)&d_out2, imgWidth * imgHeight * sizeof(unsigned char));

    //复制CPU图像数据到GPU内存指针变量
    cudaMemcpy(d_in, grayImg.data, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32,32); //定义32*32维度的block线程块，尽可能提高运算速度
    dim3 blocksPerGrid((imgWidth+threadsPerBlock.x-1)/threadsPerBlock.x,
    		           (imgHeight+threadsPerBlock.y-1)/threadsPerBlock.y); //根据不同图片尺寸大小定义最优的线程网格维度

    //算子大小
    cv::Size Element(3, 5);

    //cuda腐蚀计算
    erodeInCude<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out1, Element, imgWidth, imgHeight);
    //cuda膨胀计算
    dilateInCuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out1, Element, imgWidth, imgHeight);

    //将GPU计算结果变量赋值回主机CPU端
    cudaMemcpy(erodeImg.data, d_out1, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(dilateImg.data, d_out2, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::imshow("erode", erodeImg);
    cv::waitKey();
    cv::imshow("dilate", dilateImg);
    cv::waitKey();

    //opencv实现
    //腐蚀
    cv::Mat erodeImg_cv;
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
    erode(grayImg, erodeImg_cv, element);

    //膨胀
    cv::Mat dilateImg_cv;
    dilate(grayImg, dilateImg_cv, element);


    //释放GPU内存指针变量
    cudaFree(d_in);
    cudaFree(d_out1);
    cudaFree(d_out2);

    return 0;
}







