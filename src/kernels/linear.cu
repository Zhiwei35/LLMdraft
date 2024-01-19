#include <iostream>
#include "src/kernels/linear.h"
// TODO: when abstracted weight class, replace T with class
// all matmul cases:
// ctx qkv lienar: [num_tokens, qhiddenunits] * [qhiddenunits, hiddenunits] = {num_tokens, qkv_head_num,  head_size}
// ctx attn output linear: {num_tokens, head_num, head_size} * {q hidden units, q hidden units} = {num_tokens, q hidden units}
// self qkv linear: [bs, q hidden units] * [qhiddenunits, hiddenunits] = {bs, qkv_head_num,  head_size}}
// self attn output linear: {batch_size, q hidden_units} * [qhiddenunits, qhiddenunits] = [bs, q hiddenunits]
// lmhead linear: [bs, q hidden units] * [vocab size, q hiden units], need transpose B
// gate:[bs/token nums, q hidden units] * [q hidden units, inter size] = [bs/token nums, inter size]
// up:[bs/token nums, q hidden units] * [q hidden units, inter size] = [bs/token nums, inter size]
// fusedGateUpGemm: [bs/token nums, q hidden units] * [q hidden units, 2 * inter size] = [bs/token nums, 2 * inter size]
// down:[bs/token nums, inter size] * [inter size, q hidden units] = [bs/token nums, q hidden units]
template <typename T>
void launchLinearGemm(TensorWrapper<T> *input,
                      BaseWeight<T> &weight,
                      TensorWrapper<T> *output,
                      cublasWrapper *cublas_wrapper,
                      bool trans_a,
                      bool trans_b)
{
    int Am = weight.shape[1];
    int Ak = weight.shape[0];
    int Bk = input->shape[1];
    int Bn = input->shape[0];
    int Cm = output->shape[1];
    int Cn = output->shape[0];

    // for ctx attn and self attn qkv linear, assume [bs/token nums, qkv h ead num, head size]
    // for gate & up linear, assume weight.shape=[hidden,2*intersize], output.shape=[bs, 2, inter size]
    Cm = output->shape.size() == 3 ? output->shape[1] * output->shape[2] : output->shape[1];

    // for ctx attn output linear
    Bk = input->shape.size() == 3 ? input->shape[1] * input->shape[2] : input->shape[1];
    int lda = Am;
    int ldb = Bk;
    int ldc = Cm;
    // input length > 1 表示当前为first token的lmhead, 参考自fastllm, 去[maxlen-1,maxlen)范围的tensor参与lmhead即可，反之为second token的lmhead
    // transformers里面是不是和fastllm一样的做法还有待确认，没有在causaloutputwithpast里面找到

    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N; // for lmhead linear
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    // int offset = 0;
    // if (shared_out_buf)
    // {
    //     ONELLM_CHECK_WITH_INFO(output->shape.size() == 3, "output shape should be 3 dims, that is [2, num tokens, hidden units]");
    //     offset = output->shape[1] * output->shape[2]; // num tokes * inter size, need to modify activate kernel input shape to [2, num tokens, inter size] and buf shape
    // }
    // std::cout << "shared offset: " << offset << std::endl;
    //  std::cout << "m: " << input_lda
    //            << "n: " << n << " or " << weight_1st_dim
    //            << "k: " << weight_ldb << "\n" // 32
    //            << "weight shape: " << weight.shape[0] << "," << weight.shape[1] << "\n"
    //            << "output shape: " << output->shape[0] << "," << output->shape[1] << "\n";
    if (!trans_a && !trans_b)
    {
        ONELLM_CHECK_WITH_INFO(Ak == Bk, "2nd dim of input MUST = 1st dim of weight");
    }

    cublas_wrapper->Gemm(transA,
                         transB,
                         trans_b ? Ak : Am, // m
                         Cn,                // n, when load real weight, lmhead weight is same as pre embedding, which shape = [vocab, hidden], so here should transpose b
                         Bk,
                         weight.data,                     // A, cur_input_len is for context decoder lmhead
                         lda,                             // lda
                         input->data, // B
                         ldb,                             // ldb
                         output->data,           // C
                         ldc,                             // ldc
                         1.0f,
                         0.0f);
}

template <typename T>
void launchLinearGemmForCtxDecoderLMhead(TensorWrapper<T> *input,
                                        BaseWeight<T> &weight,
                                        TensorWrapper<T> *output,
                                        cublasWrapper *cublas_wrapper,
                                        bool trans_a,
                                        bool trans_b)
{
    int Am = weight.shape[1];
    int Ak = weight.shape[0];
    int Bk = input->shape[1];
    int Bn = input->shape[0];
    int Cm = output->shape[1];
    int Cn = output->shape[0];

    int lda = Am;
    int ldb = Bk;
    int ldc = Cm;
    // > 1 表示当前为first token的lmhead, 参考自fastllm, 去[maxlen-1,maxlen)范围的tensor参与lmhead即可，反之为second token的lmhead
    // transformers里面是不是和fastllm一样的做法还有待确认，没有在causaloutputwithpast里面找到
    // ldb = cur_input_len > 1 ? 1 : Bk; 不需要修改ldb，此时ldb为hiddenunits，seqlen维度在第二维
    // int outer = input.Count(0) / input.Count(axis); // dims[0] * strides[0] / dims[axis] * strides[axis]
    // int inputStride = input.Count(axis);
    // int outputStride = output.Count(axis);
    // int channels = input.dims[axis];
    // int inner = input.strides[axis];
    // int unitSize = (int)sizeof(T);// sizeof(T)

    // cudaMemcpy2D((void*)output.cudaData, outputStride * unitSize,
    //                                   (void*)input.cudaData + start * inner * unitSize, inputStride * unitSize,
    //                                   (cur_input_len - (cur_input_len - 1)) * inner * unitSize, outer, cudaMemcpyDeviceToDevice);// height rows of width bytes

    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N; // for lmhead linear
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    // std::cout << "shared offset: " << offset << std::endl;
    //  std::cout << "m: " << input_lda
    //            << "n: " << n << " or " << weight_1st_dim
    //            << "k: " << weight_ldb << "\n" // 32
    //            << "weight shape: " << weight.shape[0] << "," << weight.shape[1] << "\n"
    //            << "output shape: " << output->shape[0] << "," << output->shape[1] << "\n";

    cublas_wrapper->Gemm(transA,
                         transB,
                         trans_b ? Ak : Am, // m
                         Cn,                // n, when load real weight, lmhead weight is same as pre embedding, which shape = [vocab, hidden], so here should transpose b
                         Bk,
                         weight.data,                     // A, cur_input_len is for context decoder lmhead
                         lda,                             // lda
                         input->data, // B
                         ldb,                             // ldb
                         output->data,           // C
                         ldc,                             // ldc
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
    // B.T A.T = C.T
    // TODO:currently only consider trans_b
    int Bm = input1->shape[2]; // len q       // len q
    int Bk = input1->shape[3]; // head size   // len k
    int Ak = input2->shape[2]; // len k       // len k
    int An = input2->shape[3]; // head size   // head size
    int Cm = output->shape[2]; // len q       // len q
    int Cn = output->shape[3]; // len k       // head size
    int lda = An;
    int ldb = Bk; // ld should be val before transpose
    int ldc = Cn;
    int64_t strideA = Ak * An; // stride should be val after transpose
    int64_t strideB = Bm * Bk;
    int64_t strideC = Cm * Cn;
    // TODO:check 4nd dim of input = 3rd dim of weight
    // TODO:check batchCount of two matrix is equal
    int batchCount = input1->shape[0] * input1->shape[1];

    // std::cout << "calling batch gemm" << "\n";
    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublas_wrapper->stridedBatchedGemm(transA,
                                       transB,
                                       Cn,           // m
                                       Cm,           // n
                                       Bk,           // k
                                       input2->data, // A,[Bk, Bn]=[bs, head num,  head size,max k len]
                                       lda,
                                       strideA,
                                       input1->data, // B [Ak, An]=[bs, head num,  head size,max q len]
                                       ldb,
                                       strideB,
                                       output->data, // C [[bs, head num,  max k len, max q len]
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
                               bool trans_b);

template void launchLinearGemm(TensorWrapper<half> *input,
                               BaseWeight<half> &weight,
                               TensorWrapper<half> *output,
                               cublasWrapper *cublas_wrapper,
                               bool trans_a,
                               bool trans_b);

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
