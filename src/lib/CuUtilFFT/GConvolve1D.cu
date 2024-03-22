#include <CuUtilFFT/GFFT1D.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <math.h>

using namespace CuUtilFFT;

static __global__ void mGConvolve1D
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2, 
	int iCmpSize,
	cufftComplex* gResCmp
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= iCmpSize) return;
	//-----------------------
	float fRe, fIm;
	fRe = gCmp2[i].x * gCmp1[i].x + gCmp2[i].y * gCmp1[i].y;
	fIm = gCmp2[i].x * gCmp1[i].y - gCmp2[i].y * gCmp1[i].x;
	gResCmp[i].x = fRe;
	gResCmp[i].y = fIm;
}

GConvolve1D::GConvolve1D(void)
{
}

GConvolve1D::~GConvolve1D(void)
{
}

void GConvolve1D::DoIt
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	int iCmpSize,
	cufftComplex* gResCmp
)
{	dim3 aBlockDim(512, 1);
        dim3 aGridDim(1, 1);
        aGridDim.x = iCmpSize / aBlockDim.x + 1;
	mGConvolve1D<<<aGridDim, aBlockDim>>>
	(  gCmp1, gCmp2, iCmpSize, gResCmp
	);
}

