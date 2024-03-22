python_add_library(_pygctffind
    MODULE
        PyInput.cpp
        PyFolder.cpp

        ${PYGCTFFIND_INC}/PyInput.h
        ${PYGCTFFIND_INC}/PyFolder.h
)

target_link_libraries(_pygctffind
    PRIVATE
        pybind11::headers
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

target_include_directories(_pygctffind
    PUBLIC
        ${GCTFFIND_INC}
    PRIVATE
        ${LIB_INC}
        ${PYGCTFFIND_INC}
)

# The install directory is the output (wheel) directory
install(TARGETS _pygctffind DESTINATION ${CMAKE_SOURCE_DIR}/src/pyGCTFFind)