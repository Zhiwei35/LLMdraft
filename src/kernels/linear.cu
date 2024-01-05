#include <iostream>
#include "src/kernels/linear.h"
// TODO: when abstracted weight class, replace T with class
// weight * input
// weight shape = [hidden_units, hidden_units]
// input shape = [hidden_units, seqlen]
template <typename T>
void launchLinearGemm(TensorWrapper<T> *input,
                      BaseWeight<T> &weight,
                      TensorWrapper<T> *output,
                      cublasWrapper *cublas_wrapper,
                      bool trans_a,
                      bool trans_b,
                      bool shared_out_buf,
                      int cur_input_len)
{
    int input_lda = cur_input_len > 1 ? 1 : input->shape[0];
    int weight_ldb = weight.shape[0];
    int weight_1st_dim = weight.shape[0];
    int weight_2nd_dim = weight.shape[1];

    int output_ldc = input_lda;
    int n = output->shape.size() == 3 ? output->shape[2] : output->shape[1];
    cublasOperation_t transA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    int offset = 0;
    if (shared_out_buf)
    {
        ONELLM_CHECK_WITH_INFO(output->shape.size() == 3, "output shape should be 3 dims, that is [2, num tokens, hidden units]");
        offset = input_lda * output->shape[2]; // num tokes * inter size, need to modify activate kernel input shape to [2, num tokens, inter size] and buf shape
    }
    //std::cout << "shared offset: " << offset << std::endl;
    // std::cout << "m: " << input_lda
    //           << "n: " << n << " or " << weight_1st_dim
    //           << "k: " << weight_ldb << "\n" // 32
    //           << "weight shape: " << weight.shape[0] << "," << weight.shape[1] << "\n"
    //           << "output shape: " << output->shape[0] << "," << output->shape[1] << "\n";
    if (!trans_a && !trans_b)
    {
        ONELLM_CHECK_WITH_INFO(weight.shape[0] == weight_ldb, "2nd dim of input MUST = 1st dim of weight");
    }
    else if (trans_b)
    {
        if (input->shape.size() > 2)
        {
            ONELLM_CHECK_WITH_INFO(input->shape[2] == weight.shape[1], "when trans_b, 2nd dim of input MUST = 2nd dim of weight");
        }
        else
        {
            ONELLM_CHECK_WITH_INFO(input->shape[1] == weight.shape[1], "when trans_b, 2nd dim of input MUST = 2nd dim of weight");
        }
    }

    cublas_wrapper->Gemm(transA,
                         transB,
                         input_lda,                                      // m
                         trans_b ? weight_1st_dim : n,                   // n, when load real weight, lmhead weight is same as pre embedding, which shape = [vocab, hidden], so here should transpose b
                         trans_b ? weight_2nd_dim : weight_ldb,          // k
                         input->data + (cur_input_len - 1) * weight_ldb, // A, cur_input_len is for context decoder lmhead
                         input_lda,                                      // lda
                         weight.data,                                    // B
                         weight_ldb,                                     // ldb
                         output->data + offset,                          // C
                         output_ldc,                                     // ldc
                         1.0f,
                         0.0f);
}
template <typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T> *input1,
                                  TensorWrapper<T> *input2,
                                  TensorWrapper<T> *output,
                                  cublasWrapper *cublas_wrapper,
                                  bool trans_a,
                                  bool trans_b)
{
    // TODO:currently only consider trans_b
    int Am = input1->shape[2];
    int Ak = input1->shape[3];
    int Bk = input2->shape[2];
    int Bn = input2->shape[3];
    int lda = Am;
    int ldb = Bk;
    int ldc = Am;
    int64_t strideA = Am * Ak;
    int64_t strideB = Bk * Bn;
    int64_t strideC = Am * Bn;
    // TODO:check 4nd dim of input = 3rd dim of weight
    // TODO:check batchCount of two matrix is equal
    int batchCount = input1->shape[0] * input1->shape[1];

    // std::cout << "calling batch gemm" << "\n";
    cublasOperation_t transA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublas_wrapper->stridedBatchedGemm(transA,
                                       transB,
                                       Am,
                                       trans_b ? Bk : Bn,
                                       Ak,
                                       input1->data, // A
                                       lda,
                                       strideA,
                                       input2->data, // B
                                       ldb,
                                       strideB,
                                       output->data, // C
                                       ldc,
                                       strideC,
                                       batchCount,
                                       1.0f,
                                       0.0f);
    // std::cout << "called batch gemm" <<"\n";
}

template void launchLinearGemm(TensorWrapper<float> *input,
                               BaseWeight<float> &weight,
                               TensorWrapper<float> *output,
                               cublasWrapper *cublas_wrapper,
                               bool trans_a,
                               bool trans_b,
                               bool shared_out_buf,
                               int cur_input_len);

template void launchLinearGemm(TensorWrapper<half> *input,
                               BaseWeight<half> &weight,
                               TensorWrapper<half> *output,
                               cublasWrapper *cublas_wrapper,
                               bool trans_a,
                               bool trans_b,
                               bool shared_out_buf,
                               int cur_input_len);

template void launchLinearStridedBatchGemm(TensorWrapper<float> *input1,
                                           TensorWrapper<float> *input2,
                                           TensorWrapper<float> *output,
                                           cublasWrapper *cublas_wrapper,
                                           bool trans_a,
                                           bool trans_b);

template void launchLinearStridedBatchGemm(TensorWrapper<half> *input1,
                                           TensorWrapper<half> *input2,
                                           TensorWrapper<half> *output,
                                           cublasWrapper *cublas_wrapper,
                                           bool trans_a,
                                           bool trans_b);
