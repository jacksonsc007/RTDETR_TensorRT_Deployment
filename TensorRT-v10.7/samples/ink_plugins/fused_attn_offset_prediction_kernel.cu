#include <iostream>
#include "fused_attn_offset_prediction_kernel.h"

__global__ void kernel_fp32_mul(
    int8_t const * activation,
    float const* activation_scale,
    int8_t const* activation_offset,
    float const* fused_weight_scale,
    int8_t const* fused_weight_offset,
    int8_t const* fused_weight,
    float const* fused_weight_bias,
    int32_t n_rows,
    int32_t n_cols,
    int32_t nC, // number channels 
    float * fused_output
)
{
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x; // column idx
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y; // row idx

    float s_a = *activation_scale;
    int8_t a_offset = *activation_offset;
    float acc = 0.0f;
    
    // attn result
    if (i < n_rows && 0 <= j  && j < 96)
    {
        float w_bias = fused_weight_bias(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t w_offset = fused_weight_offset(0, j, 1);
            float s_w = fused_weight_scale(0, j, 1);
            float dq_w = (w - w_offset) * s_w;

            int8_t a = activation(i, p, nC);
            float dq_a = (a - a_offset) * s_a;

            acc += dq_w * dq_a;
            // #define DEBUG_KERNEL_FLOAT
            #ifdef DEBUG_KERNEL_FLOAT
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %3.4f "
                    "dq_w: %8.4f dq_a: %8.4f\n", 
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w, w_bias, n_rows, n_cols, nC, acc,
                    dq_w, dq_a
                );
            #endif
            
        }
        acc += w_bias;
        fused_output(i, j, 96) = acc;
        // if (i == 0 )
        //     printf("\e[31mRows:%d Cols:%d [%d, %d] =  %f\e[m \n", n_rows, n_cols, i, j, acc);
    }
    else if (i < n_rows && j >= 96 && j < n_cols)
    {
        fused_output += n_rows * 96;
        float w_bias = fused_weight_bias(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t w_offset = fused_weight_offset(0, j, 1);
            float s_w = fused_weight_scale(0, j, 1);
            float dq_w = (w - w_offset) * s_w;

            int8_t a = activation(i, p, nC);
            float dq_a = (a - a_offset) * s_a;

            acc += dq_w * dq_a;
            #ifdef DEBUG_KERNEL
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %3.4f "
                    "dq_w: %8.4f dq_a: %8.4f\n", 
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w, w_bias, n_rows, n_cols, nC, acc,
                    dq_w, dq_a
                );
            #endif
            
        }
        acc += w_bias;
        fused_output(i, j - 96, 192) = acc;
        
    }
}

__global__ void kernel_int8_mul(
    int8_t const * activation,
    float const* activation_scale,
    int8_t const* activation_offset,
    float const* fused_weight_scale,
    int8_t const* fused_weight_offset,
    int8_t const* fused_weight,
    float const* fused_weight_bias,
    int32_t n_rows,
    int32_t n_cols,
    int32_t nC, // number channels 
    float * fused_output
)
{
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x; // column idx
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y; // row idx

    float s_a = *activation_scale;
    int8_t a_offset = *activation_offset;
    int32_t acc_i32 = 0;
    float acc_float = 0.0f;
    
    // attn result
    if (i < n_rows && 0 <= j  && j < 96)
    {
        float s_w = fused_weight_scale(0, j, 1);
        float w_bias = fused_weight_bias(0, j, 1);
        int8_t w_offset = fused_weight_offset(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t a = activation(i, p, nC);

            int16_t i8_prod = (w - w_offset) * (a - a_offset);
            acc_i32 += i8_prod;
            // #define DEBUG_KERNEL_INT8
            #ifdef DEBUG_KERNEL_INT8
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %4d ",
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w,
                    w_bias, n_rows, n_cols, nC, acc_i32
                );
            #endif
            
        }
        acc_float = (float)acc_i32 * (s_a * s_w);
        acc_float += w_bias;
        fused_output(i, j, 96) = acc_float;
    }
    else if (i < n_rows && j >= 96 && j < n_cols)
    {
        fused_output += n_rows * 96;
        float s_w = fused_weight_scale(0, j, 1);
        float w_bias = fused_weight_bias(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t w_offset = fused_weight_offset(0, j, 1);
            int8_t a = activation(i, p, nC);

            int16_t i8_prod = (w- w_offset) * (a - a_offset);
            acc_i32 += i8_prod;
            #ifdef DEBUG_KERNEL
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %3.4f "
                    "dq_w: %8.4f dq_a: %8.4f\n", 
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w, w_bias, n_rows, n_cols, nC, acc,
                    dq_w, dq_a
                );
            #endif
            
        }
        acc_float = (float)acc_i32 * (s_a * s_w);
        acc_float += w_bias;
        fused_output(i, j - 96, 192) = acc_float;
        
    }



}

__global__ void debug_kernel(
    float const * activation,
    float const* activation_scale,
    int8_t const* activation_offset,
    float const* fused_weight_scale,
    int8_t const* fused_weight_offset,
    int8_t const* fused_weight,
    float const* fused_weight_bias,
    int32_t n_rows,
    int32_t n_cols,
    int32_t nC, // number channels 
    float * fused_output
)
{
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x; // column idx
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y; // row idx

    float s_a = *activation_scale;
    int8_t a_offset = *activation_offset;
    float acc = 0.0f;
    
    // attn result
    if (i < n_rows && 0 < j  && j < 96)
    {
        float w_bias = fused_weight_bias(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t w_offset = fused_weight_offset(0, j, 1);
            float s_w = fused_weight_scale(0, j, 1);
            float dq_w = (w - w_offset) * s_w;

            float a = activation(i, p, nC);
            float dq_a = (a - a_offset) * s_a;

            acc += dq_w * dq_a;
            #define DEBUG_KERNEL_FLOAT
            #ifdef DEBUG_KERNEL_FLOAT
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %3.4f "
                    "dq_w: %8.4f dq_a: %8.4f\n", 
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w, w_bias, n_rows, n_cols, nC, acc,
                    dq_w, dq_a
                );
            #endif
            
        }
        acc += w_bias;
        fused_output(i, j, 96) = acc;
        // if (i == 0 )
        //     printf("\e[31mRows:%d Cols:%d [%d, %d] =  %f\e[m \n", n_rows, n_cols, i, j, acc);
    }
    else if (i < n_rows && j >= 96 && j < n_cols)
    {
        fused_output += n_rows * 96;
        float w_bias = fused_weight_bias(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t w_offset = fused_weight_offset(0, j, 1);
            float s_w = fused_weight_scale(0, j, 1);
            float dq_w = (w - w_offset) * s_w;

            int8_t a = activation(i, p, nC);
            float dq_a = (a - a_offset) * s_a;

            acc += dq_w * dq_a;
            #ifdef DEBUG_KERNEL
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %3.4f "
                    "dq_w: %8.4f dq_a: %8.4f\n", 
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w, w_bias, n_rows, n_cols, nC, acc,
                    dq_w, dq_a
                );
            #endif
            
        }
        acc += w_bias;
        fused_output(i, j - 96, 192) = acc;
        
    }
}
void fused_attn_offset_prediction_impl(
    const int32_t batchsize,
    const int32_t num_queries,
    const int32_t n_channels,
    int8_t const * activation,
    float const* activation_scale,
    int8_t const* activation_offset,
    float const* fused_weight_scale,
    int8_t const* fused_weight_offset,
    int8_t const* fused_weight,
    float const* fused_weight_bias,
    int32_t n_rows,
    int32_t n_cols,
    int32_t nC, // number channels 
    float * fused_output,
    cudaStream_t stream,
    Tactic tactic
)
{
    // printf("\e[32m[fused_attn_offset_prediction_impl]\e[m \n");
    const int32_t blocksize = 32;
    const int32_t output_col = 96 + 192;
    dim3 blockDim(blocksize, blocksize, 1);
    dim3 gridDim(div_round_up(output_col, blocksize), div_round_up(num_queries, blocksize), 1);
    // print grid and block info
    // printf("\e[32m[INFO]\e[m enqueue: gridDim: (%d, %d, %d), blockDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    // printf("\e[32m[INFO]\e[m enqueue: attn_weight_scale: %f, sp_weight_scale: %f\n", attn_weight_scale[0], sp_weight_scale[0]);
    if (tactic == Tactic::int8_mul)
    {
        kernel_int8_mul<<<gridDim, blockDim, 0, stream>>>(
            activation,
            activation_scale,
            activation_offset,
            fused_weight_scale,
            fused_weight_offset,
            fused_weight,
            fused_weight_bias,
            n_rows,
            n_cols,
            nC,
            fused_output
        );
    }
    else if (tactic == Tactic::float_mul)
    {
        kernel_fp32_mul<<<gridDim, blockDim, 0, stream>>>(
            activation,
            activation_scale,
            activation_offset,
            fused_weight_scale,
            fused_weight_offset,
            fused_weight,
            fused_weight_bias,
            n_rows,
            n_cols,
            nC,
            fused_output
        );
    }
    else
    {
        printf("\e[31m[ERROR]\e[m enqueue: invalid tactic: %d\n", (int)tactic);
        std::abort();
    }
}

void debug_fused_attn_offset_prediction_impl(
    const int32_t batchsize,
    const int32_t num_queries,
    const int32_t n_channels,
    float const * activation,
    float const* activation_scale,
    int8_t const* activation_offset,
    float const* fused_weight_scale,
    int8_t const* fused_weight_offset,
    int8_t const* fused_weight,
    float const* fused_weight_bias,
    int32_t n_rows,
    int32_t n_cols,
    int32_t nC, // number channels 
    float * fused_output,
    cudaStream_t stream,
    Tactic tactic
)
{
    const int32_t blocksize = 32;
    const int32_t output_col = 96 + 192;
    dim3 blockDim(blocksize, blocksize, 1);
    dim3 gridDim(div_round_up(output_col, blocksize), div_round_up(num_queries, blocksize), 1);
    // print grid and block info
    // printf("\e[32m[INFO]\e[m enqueue: gridDim: (%d, %d, %d), blockDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    // printf("\e[32m[INFO]\e[m enqueue: attn_weight_scale: %f, sp_weight_scale: %f\n", attn_weight_scale[0], sp_weight_scale[0]);
        debug_kernel<<<gridDim, blockDim, 0, stream>>>(
            activation,
            activation_scale,
            activation_offset,
            fused_weight_scale,
            fused_weight_offset,
            fused_weight,
            fused_weight_bias,
            n_rows,
            n_cols,
            nC,
            fused_output
        );
}