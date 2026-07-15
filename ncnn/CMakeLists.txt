cmake_minimum_required(VERSION 3.20)

project(qwen3_asr_ncnn VERSION 0.1.0 LANGUAGES C CXX)

option(QWEN3_ASR_NCNN_BUILD_TESTS "Build model-independent tests" ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
    add_compile_definitions(NOMINMAX)
endif()

find_package(ncnn CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(qwen3_asr_ncnn_runtime STATIC
    src/ncnn_llm_asr.cpp
    src/ncnn_text_runtime.cpp
    src/sampling.cpp
    src/utils/rope_embed.cpp
    src/utils/tokenizer/bpe_tokenizer.cpp
)
target_include_directories(qwen3_asr_ncnn_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(qwen3_asr_ncnn_runtime
    PUBLIC
        ncnn
        nlohmann_json::nlohmann_json
)

add_executable(qwen3_asr_ncnn examples/asr_main.cpp)
target_include_directories(qwen3_asr_ncnn PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/examples")
target_link_libraries(qwen3_asr_ncnn PRIVATE qwen3_asr_ncnn_runtime)
if(WIN32)
    target_link_libraries(qwen3_asr_ncnn PRIVATE shell32)
endif()

if(QWEN3_ASR_NCNN_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(qwen3_asr_ncnn_tests tests/test_asr_layout.cpp)
        target_link_libraries(qwen3_asr_ncnn_tests PRIVATE qwen3_asr_ncnn_runtime)
        add_test(NAME qwen3_asr_ncnn_tests COMMAND qwen3_asr_ncnn_tests)
        add_test(NAME qwen3_asr_ncnn_help COMMAND qwen3_asr_ncnn --help)
    endif()
endif()
