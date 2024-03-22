add_library(
        cuutilfft
        STATIC
            GConvolve1D.cu
            GConvolve2D.cu
            GFFT1D.cu
            GFFT2D.cu
            GFFTResize2D.cu
            GLowpass1D.cu
            GLowpass2D.cu
            GShiftFFT1D.cu
            GShiftFFT2D.cu

            ${LIB_INC}/CuUtilFFT/GFFT1D.h
            ${LIB_INC}/CuUtilFFT/GFFT2D.h
)

target_include_directories(
        cuutilfft
        PUBLIC
            ${LIB_INC}
        PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(
        cuutilfft
        PROPERTIES
            CUDA_ARCHITECTURES all
)