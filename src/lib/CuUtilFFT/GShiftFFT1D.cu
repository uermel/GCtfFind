#include <CuUtilFFT/GFFT1D.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <math.h>

using namespace CuUtilFFT;

static __global__ void mGShift
(	cufftComplex* gCmp, 
	int iCmpSize,
	float fShift
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= iCmpSize) return;
	//-----------------------
	float fTemp = i * fShift;
	float fCos = cosf(fTemp);
	float fSin = sinf(fTemp);
	//-----------------------
	fTemp = fCos * gCmp[i].x - fSin * gCmp[i].y;
	gCmp[i].y = fCos * gCmp[i].y + fSin * gCmp[i].x;
	gCmp[i].x = fTemp;
}

static __global__ void mGCenter
(	cufftComplex* gCmp, 
	int iCmpSize
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= iCmpSize) return;
        //-----------------------
	int iSign = (i % 2 == 0) ? 1 : -1;
	gCmp[i].x *= iSign;
	gCmp[i].y *= iSign;
}

GShiftFFT1D::GShiftFFT1D(void)
{
	m_f2PI = (float)(8 * atan(1.0));
}

GShiftFFT1D::~GShiftFFT1D(void)
{
}

void GShiftFFT1D::DoIt
(	cufftComplex* gCmp, 
	int iCmpSize,
	float fShift
)
{	if(fShift == 0) return;
	//---------------------	
	int iN = (iCmpSize - 1) * 2;
	fShift = fShift * m_f2PI / iN;
	//----------------------------
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = iCmpSize / aBlockDim.x + 1;
	mGShift<<<aGridDim, aBlockDim>>>
	(  gCmp, iCmpSize, fShift
	);
}

void GShiftFFT1D::Center(cufftComplex* gCmp, int iCmpSize)
{
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = iCmpSize / aBlockDim.x + 1;
	mGCenter<<<aGridDim, aBlockDim>>>(gCmp, iCmpSize);
}
