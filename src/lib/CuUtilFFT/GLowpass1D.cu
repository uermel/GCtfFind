#include <CuUtilFFT/GFFT1D.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <math.h>

using namespace CuUtilFFT;

static __global__ void mGDoBFactor
(	cufftComplex* gInCmp, 
	int iCmpSize,
	float fScale,
	cufftComplex* gOutCmp
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= iCmpSize) return;
	//-----------------------
	float fFilt = expf(fScale * i * i);
	gOutCmp[i].x = gInCmp[i].x * fFilt;
	gOutCmp[i].y = gInCmp[i].y * fFilt;
}

static __global__ void mGDoCutoff
(	cufftComplex* gInCmp,
	int iCmpSize,
	float fCutoff,
	cufftComplex* gOutCmp
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= iCmpSize) return;
        //-----------------------
	float fX = i * 0.5f / (iCmpSize - 1);
	float fR = sqrtf(fX * fX) / fCutoff;
	//----------------------------------
	float fFilt = 0.0f;
	if(fR < 1)
	{	fR = 0.5f * (1.0f - cosf(3.1416f * fR));
		fFilt = 1.0f - powf(fR, 20.0f);
	}
	gOutCmp[i].x = gInCmp[i].x * fFilt;
	gOutCmp[i].y = gInCmp[i].y * fFilt;
}

GLowpass1D::GLowpass1D(void)
{
}

GLowpass1D::~GLowpass1D(void)
{
}

void GLowpass1D::DoBFactor
(	cufftComplex* gInCmp,
	cufftComplex* gOutCmp,
	int iCmpSize,
	float fBFactor
)
{	int iN = (iCmpSize - 1) * 2;
	float fScale = -fBFactor / iN / iN;
	//---------------------------------
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = iCmpSize / aBlockDim.x + 1;
	mGDoBFactor<<<aGridDim, aBlockDim>>>
	(  gInCmp, iCmpSize, fScale, gOutCmp
	);
}

cufftComplex* GLowpass1D::DoBFactor
(	cufftComplex* gCmp,
	int iCmpSize,
	float fBFactor
)
{	size_t tBytes = sizeof(cufftComplex) * iCmpSize;
	cufftComplex* gOutCmp = 0L;
	cudaMalloc(&gOutCmp, tBytes);
	this->DoBFactor(gCmp, gOutCmp, iCmpSize, fBFactor);
	return gOutCmp;
}

void GLowpass1D::DoCutoff
(	cufftComplex* gInCmp,
	cufftComplex* gOutCmp,
	int iCmpSize,
	float fCutoff
)
{	dim3 aBlockDim(512, 1);
        dim3 aGridDim(1, 1);
        aGridDim.x = iCmpSize / aBlockDim.x + 1;
	mGDoCutoff<<<aGridDim, aBlockDim>>>
	(  gInCmp, iCmpSize, fCutoff, gOutCmp
	);
}

cufftComplex* GLowpass1D::DoCutoff
(	cufftComplex* gCmp,
	int iCmpSize,
	float fCutoff
)
{	size_t tBytes = sizeof(cufftComplex) * iCmpSize;
        cufftComplex* gOutCmp = 0L;
        cudaMalloc(&gOutCmp, tBytes);
	this->DoCutoff(gCmp, gOutCmp, iCmpSize, fCutoff);
	return gOutCmp;
}

