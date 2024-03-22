#include <CuUtilFFT/GFFT2D.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace CuUtilFFT;

static __global__ void mGCrop
( 	cufftComplex* gCmpIn, 
	int iSizeInX, int iSizeInY,
  	cufftComplex* gCmp, int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
        if(y > iCmpY / 2) y = iSizeInY + (y - iCmpY);
	int j = y * iSizeInX + blockIdx.x;
	gCmp[i].x = gCmpIn[j].x;
	gCmp[i].y = gCmpIn[j].y;
}

static __global__ void mGPad
(	cufftComplex* gCmpIn, int iCmpInY,
	cufftComplex* gCmp, int iCmpX, int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y > iCmpInY) return;
	int j =  y * gridDim.x + blockIdx.x;
	//----------------------------------
	if(y > iCmpInY / 2) y = iCmpY + (y - iCmpInY);
	int i = y * iCmpX + blockIdx.x;
	gCmp[i].x = gCmpIn[j].x;
	gCmp[i].y = gCmpIn[j].y;
}

GFFTResize2D::GFFTResize2D(void)
{
}

GFFTResize2D::~GFFTResize2D(void)
{
}

void GFFTResize2D::GetNewCmpSize
(	int* piCmpSize, float fBinning,
	int* piNewSize
)
{	int aiImgSize[2] = {0};
	aiImgSize[0] = (piCmpSize[0] - 1) * 2;
	aiImgSize[1] = piCmpSize[1];
	this->GetNewImgSize(aiImgSize, fBinning, piNewSize);
	piNewSize[0] = piNewSize[0] / 2 + 1;
	piNewSize[1] = piNewSize[1];
}

void GFFTResize2D::GetNewImgSize
(	int* piImgSize, float fBinning,
	int* piNewSize
)
{	piNewSize[0] = (int)(piImgSize[0] / fBinning);
	piNewSize[1] = (int)(piImgSize[1] / fBinning);
	piNewSize[0] = piNewSize[0] / 2 * 2;
	piNewSize[1] = piNewSize[1] / 2 * 2;
}

void GFFTResize2D::DoIt
( 	cufftComplex* gCmpIn, int* piSizeIn,
  	cufftComplex* gCmpOut, int* piSizeOut
)
{	if(piSizeIn[0] > piSizeOut[0])
	{	mCrop(gCmpIn, piSizeIn, gCmpOut, piSizeOut);
	}
	else
	{	mPad(gCmpIn, piSizeIn, gCmpOut, piSizeOut);
	}
}

cufftComplex* GFFTResize2D::DoIt
(	cufftComplex* gCmpIn, int* piSizeIn,
	float fBinning, int* piSizeOut
)
{	this->GetNewCmpSize(piSizeIn, fBinning, piSizeOut);
	size_t tBytes = sizeof(cufftComplex) * piSizeOut[0] * piSizeOut[1];
	cufftComplex* gCmpOut = 0L;
	cudaMalloc(&gCmpOut, tBytes);
	this->DoIt(gCmpIn, piSizeIn, gCmpOut, piSizeOut);
	return gCmpOut;
}

void GFFTResize2D::mCrop
(	cufftComplex* gCmpIn, int* piSizeIn,
	cufftComplex* gCmpOut, int* piSizeOut
)
{	size_t tBytes = sizeof(cufftComplex) * piSizeOut[0] * piSizeOut[1];
	cudaMemset(gCmpOut, 0, tBytes);
	//-----------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piSizeOut[0], 1);
	aGridDim.y = piSizeOut[1] / aBlockDim.y + 1;
	//------------------------------------------
	mGCrop<<<aGridDim, aBlockDim>>>
	(  gCmpIn, piSizeIn[0], piSizeIn[1],
	   gCmpOut, piSizeOut[1]
	);
}

void GFFTResize2D::mPad
(	cufftComplex* gCmpIn, int* piSizeIn,
	cufftComplex* gCmpOut, int* piSizeOut
)
{	size_t tBytes = sizeof(cufftComplex) * piSizeOut[0] * piSizeOut[1];
	cudaMemset(gCmpOut, 0, tBytes);
	//-----------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piSizeIn[0], 1);
	aGridDim.y = piSizeIn[1] / aBlockDim.y + 1;
	//-----------------------------------------
	mGPad<<<aGridDim, aBlockDim>>>
	(  gCmpIn, piSizeIn[1],
	   gCmpOut, piSizeOut[0], piSizeOut[1]
	);
}
