#include <CuUtilFFT/GFFT2D.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <math.h>

using namespace CuUtilFFT;

static __global__ void mGConvolve2D
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2, 
	int iCmpY,
	cufftComplex* gResCmp
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	float fRe, fIm;
	fRe = gCmp2[i].x * gCmp1[i].x + gCmp2[i].y * gCmp1[i].y;
	fIm = gCmp2[i].x * gCmp1[i].y - gCmp2[i].y * gCmp1[i].x;
	gResCmp[i].x = fRe;
	gResCmp[i].y = fIm;
}

GConvolve2D::GConvolve2D(void)
{
}

GConvolve2D::~GConvolve2D(void)
{
}

void GConvolve2D::DoIt
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	int* piCmpSize,
	cufftComplex* gResCmp
)
{	dim3 aBlockDim(1, 512);
        dim3 aGridDim(piCmpSize[0], 1);
        aGridDim.y = piCmpSize[1] / aBlockDim.y + 1;
	mGConvolve2D<<<aGridDim, aBlockDim>>>
	(  gCmp1, gCmp2, piCmpSize[1], gResCmp
	);
}

