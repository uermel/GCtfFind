#include <PyInput.h>

namespace py = pybind11;

namespace GCTFFind
{
    PyInput::PyInput() :
        m_fKv{300.0f}, //kV
        m_fCs{2.7f},   //mm
        m_fAmpContrast{0.07f},
        m_fPixelSize{1.0f}, // A
        m_iLogSpect{0},
        m_iTileSize{512},
        m_iGpuID{0},
        m_afExtPhase{0.0f, 0.0f},
        m_afTiltRange{0.0f, 0.0f}
    {}

    PyInput::~PyInput()
    {}

    CInput *PyInput::GetInstance(void) {
        CInput* pInstance = GCTFFind::CInput::GetInstance();
        pInstance->m_fKv = m_fKv;
        pInstance->m_fCs = m_fCs;
        pInstance->m_fAmpContrast = m_fAmpContrast;
        pInstance->m_fPixelSize = m_fPixelSize;
        pInstance->m_afExtPhase[0] = m_afExtPhase[0];
        pInstance->m_afExtPhase[1] = m_afExtPhase[1];
        pInstance->m_afTiltRange[0] = m_afTiltRange[0];
        pInstance->m_afTiltRange[1] = m_afTiltRange[1];
        pInstance->m_iLogSpect = m_iLogSpect;
        pInstance->m_iTileSize = m_iTileSize;
        pInstance->m_iGpuID = m_iGpuID;

        return pInstance;
    }
}

int add(int i, int j) {
    return i + j;
}


PYBIND11_MODULE(_pygctffind, m) {
//m.def("add", &add, "A function that adds two numbers");
    py::class_<GCTFFind::PyInput>(m, "PyInput")
        .def(py::init<>())
        .def_readwrite("kV", &GCTFFind::PyInput::m_fKv)
        .def_readwrite("Cs", &GCTFFind::PyInput::m_fCs)
        .def_readwrite("AmpContrast", &GCTFFind::PyInput::m_fAmpContrast)
        .def_readwrite("PixelSize", &GCTFFind::PyInput::m_fPixelSize)
        .def_readwrite("LogSpect", &GCTFFind::PyInput::m_iLogSpect)
        .def_readwrite("TileSize", &GCTFFind::PyInput::m_iTileSize)
        .def_readwrite("GpuID", &GCTFFind::PyInput::m_iGpuID)
        .def_property_readonly("ExtPhase", [](py::object& obj)
        {
            auto& o = obj.cast<GCTFFind::PyInput&>();
            return py::array{2, o.m_afExtPhase, obj};
        }
        )
        .def_property_readonly("TiltRange", [](py::object& obj)
        {
            auto& o = obj.cast<GCTFFind::PyInput&>();
            return py::array{2, o.m_afTiltRange, obj};
        }
        );
}
