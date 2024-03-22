set(GCTFFIND_SRC
        FindCTF/CCalcBackground.cpp
        FindCTF/CCTFTheory.cpp
        FindCTF/CFindCtf1D.cpp
        FindCTF/CFindCtf2D.cpp
        FindCTF/CFindCtfBase.cpp
        FindCTF/CFindCtfHelp.cpp
        FindCTF/CFindDefocus1D.cpp
        FindCTF/CFindDefocus2D.cpp
        FindCTF/CGenAvgSpectrum.cpp
        FindCTF/CSpectrumImage.cpp
        FindCTF/GBackground1D.cu
        FindCTF/GCalcCTF1D.cu
        FindCTF/GCalcCTF2D.cu
        FindCTF/GCalcSpectrum.cu
        #FindCTF/GCalcSpectrum2D.cu
        FindCTF/GCC1D.cu
        FindCTF/GCC2D.cu
        FindCTF/GRadialAvg.cu
        FindCTF/GRemoveMean.cu
        FindCTF/GRmBackground2D.cu
        FindCTF/GRoundEdge.cu
        ${GCTFFIND_INC}/FindCTF/CFindCTFInc.h

        GCTFFindUtil/CRegSpline.cpp
        GCTFFindUtil/CParseArgs.cpp
        GCTFFindUtil/CCudaHelper.cpp
        GCTFFindUtil/CRegSpline3.cpp
        GCTFFindUtil/CRegSpline2.cpp
        GCTFFindUtil/CSaveTempMrc.cpp
        GCTFFindUtil/GAddImages.cu
        GCTFFindUtil/GCalcMoment2D.cu
        GCTFFindUtil/GNormalize2D.cu
        GCTFFindUtil/GThreshold2D.cu
        ${GCTFFIND_INC}/GCTFFindUtil/CUtilInc.h

        MrcUtil/CAsyncSaveImages.cpp
        MrcUtil/CAsyncSingleSave.cpp
        MrcUtil/CLoadImages.cpp
        MrcUtil/CSaveImages.cpp
        ${GCTFFIND_INC}/MrcUtil/CMrcUtilInc.h

        CCtfFindMain.cpp
        CCtfPackage.cpp
        CFindSeriesCtfs.cpp
        CInput.cpp
        CInputFolder.cpp
        CLoadAngFile.cpp
        CProcessMain.cpp
        CProcessThread.cpp
        CSaveCtfResults.cpp
        ${GCTFFIND_INC}/Main/CMainInc.h
)

add_executable(GCTFFind ${GCTFFIND_SRC})
add_library(gctffind ${GCTFFIND_SRC})

target_include_directories(
        GCTFFind
        PUBLIC
            ${GCTFFIND_INC}
        PRIVATE
            ${LIB_INC}
)

target_include_directories(
        gctffind
        PUBLIC
            ${GCTFFIND_INC}
        PRIVATE
            ${LIB_INC}
)

target_link_libraries(
        GCTFFind

        cuutil
        cuutilfft
        mrcfile
        util
        cuda
        cufft
        cudart
        m
        c
        ${CMAKE_THREAD_LIBS_INIT}
)

target_link_libraries(
        gctffind

        cuutil
        cuutilfft
        mrcfile
        util
        cuda
        cufft
        cudart
        m
        c
        ${CMAKE_THREAD_LIBS_INIT}
)