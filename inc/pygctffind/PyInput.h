#pragma once
#include <Main/CMainInc.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace GCTFFind
{
    class PyInput
    {
    public:
        PyInput();
        ~PyInput();

        CInput* GetInstance(void);

        float m_fKv;
        float m_fCs;
        float m_fAmpContrast;
        float m_fPixelSize;
        float m_afExtPhase[2];
        float m_afTiltRange[2];
        int m_iTileSize;
        int m_iLogSpect;
        int m_iGpuID;
    };
}
