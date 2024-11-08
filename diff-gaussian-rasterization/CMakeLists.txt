cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CXX C)

# 添加选项来控制使用哪个测试文件
option(USE_CPP_TEST "Use C++ test file instead of C test file" OFF)

# 基本设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

# 添加调试标志
add_compile_definitions(DEBUG_MODE)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g)

# 创建debug和data_dump目录
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/debug)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data_dump)

# 如果有测试数据,可以复制到构建目录
if(EXISTS ${CMAKE_SOURCE_DIR}/data_dump)
    file(COPY ${CMAKE_SOURCE_DIR}/data_dump/ DESTINATION ${CMAKE_BINARY_DIR}/data_dump)
endif()

if(USE_CPP_TEST)
    # # 查找 Python
    # find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    # # 获取 PyTorch 路径
    # execute_process(
    #     COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__), flush=True)"
    #     OUTPUT_VARIABLE TORCH_PATH
    #     OUTPUT_STRIP_TRAILING_WHITESPACE
    # )

    # # 设置 PyTorch CMAKE 路径
    # set(CMAKE_PREFIX_PATH ${TORCH_PATH})
    # set(Torch_DIR "${TORCH_PATH}/share/cmake/Torch")

    # # 查找依赖包
    # find_package(Torch REQUIRED)

    # 设置包含目录
    include_directories(
        # ${TORCH_INCLUDE_DIRS}
        # ${Python3_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/cpu_rasterizer
    )

    # 添加可执行文件
    add_executable(test
        test.cpp
        # rasterize_points.cpp
        cpu_rasterizer/forward.c
        cpu_rasterizer/rasterizer_impl.c
        cpu_rasterizer/auxiliary.c
    )

    # # 链接库
    # target_link_libraries(test "${TORCH_LIBRARIES}" ${Python3_LIBRARIES})
    
    # 添加编译定义表明使用C++测试
    target_compile_definitions(test PRIVATE USE_CPP_TEST)
    
else()
    # C版本
    add_executable(test
        test.c
        cpu_rasterizer/forward.c
        cpu_rasterizer/rasterizer_impl.c
        cpu_rasterizer/auxiliary.c
    )
    
    target_include_directories(test PRIVATE
        ${CMAKE_SOURCE_DIR}/cpu_rasterizer
    )
    
    # 添加数学库链接
    target_link_libraries(test PRIVATE m)
    
    # 添加编译定义表明使用C测试
    target_compile_definitions(test PRIVATE USE_C_TEST)
endif()

# 显示当前使用的测试版本
if(USE_CPP_TEST)
    message(STATUS "Building with C++ test file (test.cpp)")
else()
    message(STATUS "Building with C test file (test.c)")
endif()