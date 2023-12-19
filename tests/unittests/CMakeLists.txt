add_executable(embedding
    test_input_embedding.cu
)
target_link_libraries(
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor)

add_executable(rms_norm
    test_rmsnorm.cu
)
target_link_libraries(  # Libs for test_rmsnorm
    rms_norm PUBLIC
    -lcudart
    -lcudadevrt
    rmsnorm)