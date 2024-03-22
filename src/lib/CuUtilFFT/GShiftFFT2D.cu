#include <CuUtilFFT/GFFT2D.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <math.h>

using namespace CuUtilFFT;

static __global__ void mGShift
(	cufftComplex* gCmp, int iCmpY,
	float fShiftX, float fShiftY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	if(y > iCmpY / 2) y -= iCmpY;
	float fTemp = blockIdx.x * fShiftX + y * fShiftY;
	float fCos = cosf(fTemp);
	float fSin = sinf(fTemp);
	//-----------------------
	fTemp = fCos * gCmp[i].x - fSin * gCmp[i].y;
	gCmp[i].y = fCos * gCmp[i].y + fSin * gCmp[i].x;
	gCmp[i].x = fTemp;
}

static __global__ void mGCenter
(	cufftComplex* gCmp, int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
        //---------------------------------
	int iSign = ((blockIdx.x + y) % 2 == 0) ? 1 : -1;
	gCmp[i].x *= iSign;
	gCmp[i].y *= iSign;
}

GShiftFFT2D::GShiftFFT2D(void)
{
	m_f2PI = (float)(8 * atan(1.0));
}

GShiftFFT2D::~GShiftFFT2D(void)
{
}

void GShiftFFT2D::DoIt
(	cufftComplex* gCmp, int* piCmpSize,
	float* pfShift
)
{	if(pfShift == 0L) return;
	this->DoIt(gCmp, piCmpSize, pfShift[0], pfShift[1]);
}

void GShiftFFT2D::DoIt
(	cufftComplex* gCmp, int* piCmpSize,
	float fShiftX, float fShiftY
)
{	if(fShiftX == 0 && fShiftY == 0) return;
	//--------------------------------------	
	int iNx = (piCmpSize[0] - 1) * 2;
	fShiftX = fShiftX * m_f2PI / iNx;
	fShiftY = fShiftY * m_f2PI / piCmpSize[1];
	//----------------------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piCmpSize[0], 1);
	aGridDim.y = piCmpSize[1] / aBlockDim.y + 1;
	mGShift<<<aGridDim, aBlockDim>>>
	(  gCmp, piCmpSize[1],
	   fShiftX, fShiftY
	);
}

void GShiftFFT2D::Center(cufftComplex* gCmp, int* piCmpSize)
{
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piCmpSize[0], 1);
	aGridDim.y = piCmpSize[1] / aBlockDim.y + 1;
	mGCenter<<<aGridDim, aBlockDim>>>(gCmp, piCmpSize[1]);
}
