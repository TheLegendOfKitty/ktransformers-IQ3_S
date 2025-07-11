/*
 * @Description  :  
 * @Author       : Azure-Tang, Boxin Zhang
 * @Date         : 2024-07-25 13:38:30
 * @Version      : 0.2.2
 * Adapted from https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c
 * Copyright (c) 2023-2024 The ggml authors
 * Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cstdint>
#include <c10/cuda/CUDAGuard.h>

#ifdef __HIP_PLATFORM_AMD__
typedef __hip_bfloat16 nv_bfloat16;
#endif

__global__ void dequantize_q8_0_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id = global_idx; block_id < num_blocks; block_id += blockDim.x * gridDim.x){
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);
        const int8_t* cur_block = data + block_id * blk_size;
        float scale = __half2float(*((half*)cur_block));
        cur_block += 2;
        for (int i = 0; i < ele_per_blk; i++){
            output_blk[i] = scale * cur_block[i];
        }
    }
}

__global__ void dequantize_q8_0_fp16_kernel(const int8_t* data, __half* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id = global_idx; block_id < num_blocks; block_id += blockDim.x * gridDim.x) {
        __half* __restrict__ output_blk = (__half*)(output + block_id * ele_per_blk);
        const int8_t* cur_block = data + block_id * blk_size;
        float scale = __half2float(*((half*)cur_block));
        cur_block += 2;
        for (int i = 0; i < ele_per_blk; i++) {
            output_blk[i] = __float2half(scale * cur_block[i]);
        }
    }
}

__global__ void dequantize_q8_0_bf16_kernel(const int8_t* data, nv_bfloat16* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id = global_idx; block_id < num_blocks; block_id += blockDim.x * gridDim.x) {
        nv_bfloat16* __restrict__ output_blk = (nv_bfloat16*)(output + block_id * ele_per_blk);
        const int8_t* cur_block = data + block_id * blk_size;
        float scale = __half2float(*((half*)cur_block));
        cur_block += 2;
        for (int i = 0; i < ele_per_blk; i++) {
            output_blk[i] = __float2bfloat16(scale * cur_block[i]);
        }
    }
}

// __device__ void get_scale_min_k4(int j, const uint8_t * __restrict__ q, uint8_t * __restrict__ d, uint8_t * __restrict__ m) {
__device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t * __restrict__ d, uint8_t * __restrict__ m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

__global__ void dequantize_q2_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+= blockDim.x * gridDim.x){
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);

        const float d   = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 80)));
        const float min = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 82)));

        const uint8_t * __restrict__ q = (uint8_t*)(data + block_id * blk_size + 16);

        int is = 0;
        float dl, ml;

        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t* scales = (uint8_t*)(data + block_id * blk_size + (is++));
                uint8_t sc = *scales;
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                scales = (uint8_t*)(data + block_id * blk_size + (is++));
                sc = *scales;

                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

__global__ void dequantize_q2_k_fp16_kernel(const int8_t* data, __half* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+= blockDim.x * gridDim.x){
        __half* __restrict__ output_blk = (__half*)(output + block_id * ele_per_blk);

        const float d   = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 80)));
        const float min = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 82)));

        const uint8_t * __restrict__ q = (uint8_t*)(data + block_id * blk_size + 16);

        int is = 0;
        float dl, ml;

        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t* scales = (uint8_t*)(data + block_id * blk_size + (is++));
                uint8_t sc = *scales;
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ = __float2half(dl * ((int8_t)((q[l] >> shift) & 3)) - ml);

                scales = (uint8_t*)(data + block_id * blk_size + (is++));
                sc = *scales;

                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ = __float2half(dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml);

                shift += 2;
            }
            q += 32;
        }
    }
}

__global__ void dequantize_q2_k_bf16_kernel(const int8_t* data, nv_bfloat16* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+= blockDim.x * gridDim.x){
        nv_bfloat16* __restrict__ output_blk = (nv_bfloat16*)(output + block_id * ele_per_blk);

        const float d   = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 80)));
        const float min = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 82)));

        const uint8_t * __restrict__ q = (uint8_t*)(data + block_id * blk_size + 16);

        int is = 0;
        float dl, ml;

        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t* scales = (uint8_t*)(data + block_id * blk_size + (is++));
                uint8_t sc = *scales;
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ = __float2bfloat16(dl * ((int8_t)((q[l] >> shift) & 3)) - ml);

                scales = (uint8_t*)(data + block_id * blk_size + (is++));
                sc = *scales;

                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ = __float2bfloat16(dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml);

                shift += 2;
            }
            q += 32;
        }
    }
}

__global__ void dequantize_q3_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;    
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+= blockDim.x * gridDim.x){
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);

        uint32_t aux[4];
        const int8_t * scales = (const int8_t*)aux;
        const float d_all = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 108)));

        const uint8_t * __restrict__ q  = (uint8_t*)(data + block_id * blk_size + 32);
        const uint8_t * __restrict__ hm = (uint8_t*)(data + block_id * blk_size + 0);
        uint8_t m = 1;


        uint8_t* block_scales = (uint8_t*)(data + block_id * blk_size + 96);

        for (int i = 0; i < 3; i++) {  
            aux[i] = 0;  
            for (int j = 0; j < 4; j++) {  
                aux[i] |= ((uint32_t)block_scales[i * 4 + j]) << (j * 8);
            }
        }

        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

__global__ void dequantize_q3_k_fp16_kernel(const int8_t* data, __half* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;    
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+= blockDim.x * gridDim.x){
        __half* __restrict__ output_blk = (__half*)(output + block_id * ele_per_blk);

        uint32_t aux[4];
        const int8_t * scales = (const int8_t*)aux;
        const float d_all = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 108)));

        const uint8_t * __restrict__ q  = (uint8_t*)(data + block_id * blk_size + 32);
        const uint8_t * __restrict__ hm = (uint8_t*)(data + block_id * blk_size + 0);
        uint8_t m = 1;


        uint8_t* block_scales = (uint8_t*)(data + block_id * blk_size + 96);

        for (int i = 0; i < 3; i++) {  
            aux[i] = 0;  
            for (int j = 0; j < 4; j++) {  
                aux[i] |= ((uint32_t)block_scales[i * 4 + j]) << (j * 8);
            }
        }

        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ = __float2half(dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4)));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ = __float2half(dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4)));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

__global__ void dequantize_q3_k_bf16_kernel(const int8_t* data, nv_bfloat16* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;    
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+= blockDim.x * gridDim.x){
        nv_bfloat16* __restrict__ output_blk = (nv_bfloat16*)(output + block_id * ele_per_blk);

        uint32_t aux[4];
        const int8_t * scales = (const int8_t*)aux;
        const float d_all = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 108)));

        const uint8_t * __restrict__ q  = (uint8_t*)(data + block_id * blk_size + 32);
        const uint8_t * __restrict__ hm = (uint8_t*)(data + block_id * blk_size + 0);
        uint8_t m = 1;


        uint8_t* block_scales = (uint8_t*)(data + block_id * blk_size + 96);

        for (int i = 0; i < 3; i++) {  
            aux[i] = 0;  
            for (int j = 0; j < 4; j++) {  
                aux[i] |= ((uint32_t)block_scales[i * 4 + j]) << (j * 8);
            }
        }

        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ = __float2bfloat16(dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4)));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ = __float2bfloat16(dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4)));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}


__global__ void dequantize_q4_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+=blockDim.x * gridDim.x){
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);
        // const uint8_t * q = data[i].qs;
        const uint8_t * q = (uint8_t*)(data + block_id * 144 + 16);

        const float d   = __half2float(*(reinterpret_cast<const half*>(data + block_id * 144 + 0)));
        const float min = __half2float(*(reinterpret_cast<const half*>(data + block_id * 144 + 2)));
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < ele_per_blk; j += 64) {
            uint8_t* scales = (uint8_t*)(data + block_id * 144 + 4);
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *output_blk++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

__global__ void dequantize_q4_k_fp16_kernel(const int8_t* data, __half* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+=blockDim.x * gridDim.x){
        __half* __restrict__ output_blk = (__half*)(output + block_id * ele_per_blk);
        // const uint8_t * q = data[i].qs;
        const uint8_t * q = (uint8_t*)(data + block_id * 144 + 16);

        const float d   = __half2float(*(reinterpret_cast<const half*>(data + block_id * 144 + 0)));
        const float min = __half2float(*(reinterpret_cast<const half*>(data + block_id * 144 + 2)));
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < ele_per_blk; j += 64) {
            uint8_t* scales = (uint8_t*)(data + block_id * 144 + 4);
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ = __float2half(d1 * (q[l] & 0xF) - m1);
            for (int l = 0; l < 32; ++l) *output_blk++ = __float2half(d2 * (q[l]  >> 4) - m2);
            q += 32; is += 2;
        }
    }
}

__global__ void dequantize_q4_k_bf16_kernel(const int8_t* data, nv_bfloat16* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+=blockDim.x * gridDim.x){
        nv_bfloat16* __restrict__ output_blk = (nv_bfloat16*)(output + block_id * ele_per_blk);
        // const uint8_t * q = data[i].qs;
        const uint8_t * q = (uint8_t*)(data + block_id * 144 + 16);

        const float d   = __half2float(*(reinterpret_cast<const half*>(data + block_id * 144 + 0)));
        const float min = __half2float(*(reinterpret_cast<const half*>(data + block_id * 144 + 2)));
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < ele_per_blk; j += 64) {
            uint8_t* scales = (uint8_t*)(data + block_id * 144 + 4);
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ = __float2bfloat16(d1 * (q[l] & 0xF) - m1);
            for (int l = 0; l < 32; ++l) *output_blk++ = __float2bfloat16(d2 * (q[l]  >> 4) - m2);
            q += 32; is += 2;
        }
    }
}

__global__ void dequantize_q5_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id = global_idx; block_id < num_blocks; block_id += blockDim.x * gridDim.x){
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);

        const float d   = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 0)));
        const float min = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 2)));

        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 16);
        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size + 48);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        uint8_t* scales = (uint8_t*)(data + block_id * blk_size + 4);

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *output_blk++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

__global__ void dequantize_q5_k_fp16_kernel(const int8_t* data, __half* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id = global_idx; block_id < num_blocks; block_id += blockDim.x * gridDim.x){
        __half* __restrict__ output_blk = (__half*)(output + block_id * ele_per_blk);

        const float d   = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 0)));
        const float min = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 2)));

        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 16);
        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size + 48);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        uint8_t* scales = (uint8_t*)(data + block_id * blk_size + 4);

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ = __float2half(d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1);
            for (int l = 0; l < 32; ++l) *output_blk++ = __float2half(d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2);
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

__global__ void dequantize_q5_k_bf16_kernel(const int8_t* data, nv_bfloat16* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id = global_idx; block_id < num_blocks; block_id += blockDim.x * gridDim.x){
        nv_bfloat16* __restrict__ output_blk = (nv_bfloat16*)(output + block_id * ele_per_blk);

        const float d   = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 0)));
        const float min = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 2)));

        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 16);
        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size + 48);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        uint8_t* scales = (uint8_t*)(data + block_id * blk_size + 4);

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ = __float2bfloat16(d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1);
            for (int l = 0; l < 32; ++l) *output_blk++ = __float2bfloat16(d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2);
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

__global__ void dequantize_q6_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long  block_id=global_idx; block_id<num_blocks;block_id+=blockDim.x * gridDim.x){
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);
        const float d = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 208)));

        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size);
        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 128);
        const int8_t  * __restrict__ sc = (int8_t*)(data + block_id * blk_size + 192);


        for (int n = 0; n < ele_per_blk; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                output_blk[l +  0] = d * sc[is + 0] * q1;
                output_blk[l + 32] = d * sc[is + 2] * q2;
                output_blk[l + 64] = d * sc[is + 4] * q3;
                output_blk[l + 96] = d * sc[is + 6] * q4;
            }
            output_blk += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

__global__ void dequantize_q6_k_fp16_kernel(const int8_t* data, __half* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long  block_id=global_idx; block_id<num_blocks;block_id+=blockDim.x * gridDim.x){
        __half* __restrict__ output_blk = (__half*)(output + block_id * ele_per_blk);
        const float d = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 208)));

        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size);
        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 128);
        const int8_t  * __restrict__ sc = (int8_t*)(data + block_id * blk_size + 192);


        for (int n = 0; n < ele_per_blk; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                output_blk[l +  0] = __float2half(d * sc[is + 0] * q1);
                output_blk[l + 32] = __float2half(d * sc[is + 2] * q2);
                output_blk[l + 64] = __float2half(d * sc[is + 4] * q3);
                output_blk[l + 96] = __float2half(d * sc[is + 6] * q4);
            }
            output_blk += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

__global__ void dequantize_q6_k_bf16_kernel(const int8_t* data, nv_bfloat16* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long  block_id=global_idx; block_id<num_blocks;block_id+=blockDim.x * gridDim.x){
        nv_bfloat16* __restrict__ output_blk = (nv_bfloat16*)(output + block_id * ele_per_blk);
        const float d = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size + 208)));

        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size);
        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 128);
        const int8_t  * __restrict__ sc = (int8_t*)(data + block_id * blk_size + 192);


        for (int n = 0; n < ele_per_blk; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                output_blk[l +  0] = __float2bfloat16(d * sc[is + 0] * q1);
                output_blk[l + 32] = __float2bfloat16(d * sc[is + 2] * q2);
                output_blk[l + 64] = __float2bfloat16(d * sc[is + 4] * q3);
                output_blk[l + 96] = __float2bfloat16(d * sc[is + 6] * q4);
            }
            output_blk += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static constexpr __device__ int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static constexpr __device__ uint32_t iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

static constexpr __device__ uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

__global__ void dequantize_iq4_xs_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+=blockDim.x * gridDim.x) {
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);
        const float d = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size)));
        const uint16_t scales_h = *(reinterpret_cast<const uint16_t*>(data + block_id * blk_size + 2));
        const uint8_t* scales_l = (uint8_t*)(data + block_id * blk_size + 2 + 2);
        const uint8_t* qs = (uint8_t*)(data + block_id * blk_size + 2 + 2 + 4);

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                output_blk[j + 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                output_blk[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            output_blk += 32;
            qs += 16;
        }
    }
}

__global__ void dequantize_iq4_xs_fp16_kernel(const int8_t* data, __half* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+=blockDim.x * gridDim.x) {
        __half* __restrict__ output_blk = (__half*)(output + block_id * ele_per_blk);
        const float d = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size)));
        const uint16_t scales_h = *(reinterpret_cast<const uint16_t*>(data + block_id * blk_size + 2));
        const uint8_t* scales_l = (uint8_t*)(data + block_id * blk_size + 2 + 2);
        const uint8_t* qs = (uint8_t*)(data + block_id * blk_size + 2 + 2 + 4);

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                output_blk[j + 0] = __float2half(dl * kvalues_iq4nl[qs[j] & 0xf]);
                output_blk[j + 16] = __float2half(dl * kvalues_iq4nl[qs[j] >> 4]);
            }
            output_blk += 32;
            qs += 16;
        }
    }
}

__global__ void dequantize_iq4_xs_bf16_kernel(const int8_t* data, nv_bfloat16* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+=blockDim.x * gridDim.x) {
        nv_bfloat16* __restrict__ output_blk = (nv_bfloat16*)(output + block_id * ele_per_blk);
        const float d = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size)));
        const uint16_t scales_h = *(reinterpret_cast<const uint16_t*>(data + block_id * blk_size + 2));
        const uint8_t* scales_l = (uint8_t*)(data + block_id * blk_size + 2 + 2);
        const uint8_t* qs = (uint8_t*)(data + block_id * blk_size + 2 + 2 + 4);

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                output_blk[j + 0] = __float2bfloat16(dl * kvalues_iq4nl[qs[j] & 0xf]);
                output_blk[j + 16] = __float2bfloat16(dl * kvalues_iq4nl[qs[j] >> 4]);
            }
            output_blk += 32;
            qs += 16;
        }
    }
}

__global__ void dequantize_iq3_s_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+=blockDim.x * gridDim.x) {
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);
        const float d = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size)));
        const uint8_t* qs = (uint8_t*)(data + block_id * blk_size + 2);
        const uint8_t* qh = (uint8_t*)(data + block_id * blk_size + 2 + 64);
        const uint8_t* signs = (uint8_t*)(data + block_id * blk_size + 2 + 64 + 8);
        const uint8_t* scales = (uint8_t*)(data + block_id * blk_size + 2 + 64 + 8 + 32);

        int y_offset = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1 + 2 * (scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2 * (scales[ib32/2] >> 4));
            
            // Process first 32 elements with db1
            int qs_offset = ib32 * 8;
            int signs_offset = ib32 * 4;
            int qh_idx = ib32;
            
            for (int l = 0; l < 4; ++l) {
                // Construct grid indices using 8 bits from qs + 1 high bit from qh
                int idx1 = qs[qs_offset + 2*l + 0] | ((qh[qh_idx] << (8-2*l)) & 256);
                int idx2 = qs[qs_offset + 2*l + 1] | ((qh[qh_idx] << (7-2*l)) & 256);
                
                // Get 4-element vectors from grid
                uint32_t grid1 = iq3s_grid[idx1];
                uint32_t grid2 = iq3s_grid[idx2];
                uint8_t* grid1_bytes = (uint8_t*)&grid1;
                uint8_t* grid2_bytes = (uint8_t*)&grid2;
                
                // Apply signs and scaling
                uint8_t sign_byte = signs[signs_offset + l];
                for (int j = 0; j < 4; ++j) {
                    float sign1 = (sign_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    float sign2 = (sign_byte & kmask_iq2xs[j+4]) ? -1.0f : 1.0f;
                    
                    output_blk[y_offset + j] = db1 * grid1_bytes[j] * sign1;
                    output_blk[y_offset + j + 4] = db1 * grid2_bytes[j] * sign2;
                }
                y_offset += 8;
            }
            
            // Process second 32 elements with db2
            qs_offset += 8;
            signs_offset += 4;
            qh_idx += 1;
            
            for (int l = 0; l < 4; ++l) {
                // Construct grid indices
                int idx1 = qs[qs_offset + 2*l + 0] | ((qh[qh_idx] << (8-2*l)) & 256);
                int idx2 = qs[qs_offset + 2*l + 1] | ((qh[qh_idx] << (7-2*l)) & 256);
                
                // Get 4-element vectors from grid
                uint32_t grid1 = iq3s_grid[idx1];
                uint32_t grid2 = iq3s_grid[idx2];
                uint8_t* grid1_bytes = (uint8_t*)&grid1;
                uint8_t* grid2_bytes = (uint8_t*)&grid2;
                
                // Apply signs and scaling
                uint8_t sign_byte = signs[signs_offset + l];
                for (int j = 0; j < 4; ++j) {
                    float sign1 = (sign_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    float sign2 = (sign_byte & kmask_iq2xs[j+4]) ? -1.0f : 1.0f;
                    
                    output_blk[y_offset + j] = db2 * grid1_bytes[j] * sign1;
                    output_blk[y_offset + j + 4] = db2 * grid2_bytes[j] * sign2;
                }
                y_offset += 8;
            }
        }
    }
}

__global__ void dequantize_iq3_s_fp16_kernel(const int8_t* data, __half* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+=blockDim.x * gridDim.x) {
        __half* __restrict__ output_blk = (__half*)(output + block_id * ele_per_blk);
        const float d = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size)));
        const uint8_t* qs = (uint8_t*)(data + block_id * blk_size + 2);
        const uint8_t* qh = (uint8_t*)(data + block_id * blk_size + 2 + 64);
        const uint8_t* signs = (uint8_t*)(data + block_id * blk_size + 2 + 64 + 8);
        const uint8_t* scales = (uint8_t*)(data + block_id * blk_size + 2 + 64 + 8 + 32);

        int y_offset = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1 + 2 * (scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2 * (scales[ib32/2] >> 4));
            
            // Process first 32 elements with db1
            int qs_offset = ib32 * 8;
            int signs_offset = ib32 * 4;
            int qh_idx = ib32;
            
            for (int l = 0; l < 4; ++l) {
                // Construct grid indices using 8 bits from qs + 1 high bit from qh
                int idx1 = qs[qs_offset + 2*l + 0] | ((qh[qh_idx] << (8-2*l)) & 256);
                int idx2 = qs[qs_offset + 2*l + 1] | ((qh[qh_idx] << (7-2*l)) & 256);
                
                // Get 4-element vectors from grid
                uint32_t grid1 = iq3s_grid[idx1];
                uint32_t grid2 = iq3s_grid[idx2];
                uint8_t* grid1_bytes = (uint8_t*)&grid1;
                uint8_t* grid2_bytes = (uint8_t*)&grid2;
                
                // Apply signs and scaling
                uint8_t sign_byte = signs[signs_offset + l];
                for (int j = 0; j < 4; ++j) {
                    float sign1 = (sign_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    float sign2 = (sign_byte & kmask_iq2xs[j+4]) ? -1.0f : 1.0f;
                    
                    output_blk[y_offset + j] = __float2half(db1 * grid1_bytes[j] * sign1);
                    output_blk[y_offset + j + 4] = __float2half(db1 * grid2_bytes[j] * sign2);
                }
                y_offset += 8;
            }
            
            // Process second 32 elements with db2
            qs_offset += 8;
            signs_offset += 4;
            qh_idx += 1;
            
            for (int l = 0; l < 4; ++l) {
                // Construct grid indices
                int idx1 = qs[qs_offset + 2*l + 0] | ((qh[qh_idx] << (8-2*l)) & 256);
                int idx2 = qs[qs_offset + 2*l + 1] | ((qh[qh_idx] << (7-2*l)) & 256);
                
                // Get 4-element vectors from grid
                uint32_t grid1 = iq3s_grid[idx1];
                uint32_t grid2 = iq3s_grid[idx2];
                uint8_t* grid1_bytes = (uint8_t*)&grid1;
                uint8_t* grid2_bytes = (uint8_t*)&grid2;
                
                // Apply signs and scaling
                uint8_t sign_byte = signs[signs_offset + l];
                for (int j = 0; j < 4; ++j) {
                    float sign1 = (sign_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    float sign2 = (sign_byte & kmask_iq2xs[j+4]) ? -1.0f : 1.0f;
                    
                    output_blk[y_offset + j] = __float2half(db2 * grid1_bytes[j] * sign1);
                    output_blk[y_offset + j + 4] = __float2half(db2 * grid2_bytes[j] * sign2);
                }
                y_offset += 8;
            }
        }
    }
}

__global__ void dequantize_iq3_s_bf16_kernel(const int8_t* data, nv_bfloat16* output, const int blk_size, const int ele_per_blk, const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id=global_idx; block_id<num_blocks; block_id+=blockDim.x * gridDim.x) {
        nv_bfloat16* __restrict__ output_blk = (nv_bfloat16*)(output + block_id * ele_per_blk);
        const float d = __half2float(*(reinterpret_cast<const half*>(data + block_id * blk_size)));
        const uint8_t* qs = (uint8_t*)(data + block_id * blk_size + 2);
        const uint8_t* qh = (uint8_t*)(data + block_id * blk_size + 2 + 64);
        const uint8_t* signs = (uint8_t*)(data + block_id * blk_size + 2 + 64 + 8);
        const uint8_t* scales = (uint8_t*)(data + block_id * blk_size + 2 + 64 + 8 + 32);

        int y_offset = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1 + 2 * (scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2 * (scales[ib32/2] >> 4));
            
            // Process first 32 elements with db1
            int qs_offset = ib32 * 8;
            int signs_offset = ib32 * 4;
            int qh_idx = ib32;
            
            for (int l = 0; l < 4; ++l) {
                // Construct grid indices using 8 bits from qs + 1 high bit from qh
                int idx1 = qs[qs_offset + 2*l + 0] | ((qh[qh_idx] << (8-2*l)) & 256);
                int idx2 = qs[qs_offset + 2*l + 1] | ((qh[qh_idx] << (7-2*l)) & 256);
                
                // Get 4-element vectors from grid
                uint32_t grid1 = iq3s_grid[idx1];
                uint32_t grid2 = iq3s_grid[idx2];
                uint8_t* grid1_bytes = (uint8_t*)&grid1;
                uint8_t* grid2_bytes = (uint8_t*)&grid2;
                
                // Apply signs and scaling
                uint8_t sign_byte = signs[signs_offset + l];
                for (int j = 0; j < 4; ++j) {
                    float sign1 = (sign_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    float sign2 = (sign_byte & kmask_iq2xs[j+4]) ? -1.0f : 1.0f;
                    
                    output_blk[y_offset + j] = __float2bfloat16(db1 * grid1_bytes[j] * sign1);
                    output_blk[y_offset + j + 4] = __float2bfloat16(db1 * grid2_bytes[j] * sign2);
                }
                y_offset += 8;
            }
            
            // Process second 32 elements with db2
            qs_offset += 8;
            signs_offset += 4;
            qh_idx += 1;
            
            for (int l = 0; l < 4; ++l) {
                // Construct grid indices
                int idx1 = qs[qs_offset + 2*l + 0] | ((qh[qh_idx] << (8-2*l)) & 256);
                int idx2 = qs[qs_offset + 2*l + 1] | ((qh[qh_idx] << (7-2*l)) & 256);
                
                // Get 4-element vectors from grid
                uint32_t grid1 = iq3s_grid[idx1];
                uint32_t grid2 = iq3s_grid[idx2];
                uint8_t* grid1_bytes = (uint8_t*)&grid1;
                uint8_t* grid2_bytes = (uint8_t*)&grid2;
                
                // Apply signs and scaling
                uint8_t sign_byte = signs[signs_offset + l];
                for (int j = 0; j < 4; ++j) {
                    float sign1 = (sign_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    float sign2 = (sign_byte & kmask_iq2xs[j+4]) ? -1.0f : 1.0f;
                    
                    output_blk[y_offset + j] = __float2bfloat16(db2 * grid1_bytes[j] * sign1);
                    output_blk[y_offset + j + 4] = __float2bfloat16(db2 * grid2_bytes[j] * sign2);
                }
                y_offset += 8;
            }
        }
    }
}

torch::Tensor dequantize_iq3_s(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype) {
    int num_blocks = num_bytes / blk_size;
    const at::cuda::OptionalCUDAGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({ num_bytes }, options);

    cudaMemcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes, cudaMemcpyHostToDevice);

    if (target_dtype == torch::kFloat32) {
        auto output_gpu = torch::empty({ num_blocks * ele_per_blk }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        
        dim3 block_size(256);
        dim3 grid_size((num_blocks + 255) / 256);
        dequantize_iq3_s_fp32_kernel<<<grid_size, block_size>>>(data_gpu.data_ptr<int8_t>(), output_gpu.data_ptr<float>(), blk_size, ele_per_blk, num_blocks);
        
        return output_gpu;
    }
    else if (target_dtype == torch::kFloat16) {
        auto output_gpu = torch::empty({ num_blocks * ele_per_blk }, torch::TensorOptions().dtype(torch::kFloat16).device(device));
        
        dim3 block_size(256);
        dim3 grid_size((num_blocks + 255) / 256);
        dequantize_iq3_s_fp16_kernel<<<grid_size, block_size>>>(data_gpu.data_ptr<int8_t>(), output_gpu.data_ptr<__half>(), blk_size, ele_per_blk, num_blocks);
        
        return output_gpu;
    }
    else if (target_dtype == torch::kBFloat16) {
        auto output_gpu = torch::empty({ num_blocks * ele_per_blk }, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
        
        dim3 block_size(256);
        dim3 grid_size((num_blocks + 255) / 256);
        dequantize_iq3_s_bf16_kernel<<<grid_size, block_size>>>(data_gpu.data_ptr<int8_t>(), output_gpu.data_ptr<nv_bfloat16>(), blk_size, ele_per_blk, num_blocks);
        
        return output_gpu;
    }
    else {
        throw std::runtime_error("Unsupported target dtype for IQ3_S dequantization");
    }
}

torch::Tensor dequantize_q8_0(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype) {
    int num_blocks = num_bytes / blk_size;
    const at::cuda::OptionalCUDAGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({ num_bytes }, options);

    cudaMemcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes, cudaMemcpyHostToDevice);
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({ num_blocks, 32 }, torch::dtype(target_dtype).device(device));

    switch (target_dtype) {
        case torch::kFloat16:
            dequantize_q8_0_fp16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (__half*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kBFloat16:
            dequantize_q8_0_bf16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (nv_bfloat16*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kFloat32:
            dequantize_q8_0_fp32_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), output.data_ptr<float>(), blk_size, ele_per_blk, num_blocks);
            break;
        default:
            printf("target type not support\n");
            exit(0);
    }

    cudaDeviceSynchronize();
    return output;
}


torch::Tensor dequantize_q6_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype) {
    // data.numel%blk_size should be 0, else raise err
    int num_blocks = num_bytes / blk_size;

    const at::cuda::OptionalCUDAGuard device_guard(device);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    cudaMemcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes, cudaMemcpyHostToDevice);
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

    switch (target_dtype) {
        case torch::kFloat16:
            dequantize_q6_k_fp16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (__half*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kBFloat16:
            dequantize_q6_k_bf16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (nv_bfloat16*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kFloat32:
            dequantize_q6_k_fp32_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), output.data_ptr<float>(), blk_size, ele_per_blk, num_blocks);
            break;
        default:
            printf("target type not support\n");
            exit(0);
    }
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor dequantize_q5_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype) {
    int num_blocks = num_bytes / blk_size;
    const at::cuda::OptionalCUDAGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    cudaMemcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes, cudaMemcpyHostToDevice);
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

    switch (target_dtype) {
        case torch::kFloat16:
            dequantize_q5_k_fp16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (__half*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kBFloat16:
            dequantize_q5_k_bf16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (nv_bfloat16*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kFloat32:
            dequantize_q5_k_fp32_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), output.data_ptr<float>(), blk_size, ele_per_blk, num_blocks);
            break;
        default:
            printf("target type not support\n");
            exit(0);
    }
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor dequantize_q4_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype) {
    // data.numel%blk_size should be 0, else raise err
    int num_blocks = num_bytes / blk_size;
    const at::cuda::OptionalCUDAGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    cudaMemcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes, cudaMemcpyHostToDevice);
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

    switch (target_dtype) {
        case torch::kFloat16:
            dequantize_q4_k_fp16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (__half*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kBFloat16:
            dequantize_q4_k_bf16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (nv_bfloat16*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kFloat32:
            dequantize_q4_k_fp32_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), output.data_ptr<float>(), blk_size, ele_per_blk, num_blocks);
            break;
        default:
            printf("target type not support\n");
            exit(0);
    }
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor dequantize_q3_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype) {
    int num_blocks = num_bytes / blk_size;
    const at::cuda::OptionalCUDAGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    cudaMemcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes, cudaMemcpyHostToDevice);
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

    switch (target_dtype) {
        case torch::kFloat16:
            dequantize_q3_k_fp16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (__half*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kBFloat16:
            dequantize_q3_k_bf16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (nv_bfloat16*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kFloat32:
            dequantize_q3_k_fp32_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), output.data_ptr<float>(), blk_size, ele_per_blk, num_blocks);
            break;
        default:
            printf("target type not support\n");
            exit(0);
    }
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor dequantize_q2_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype) {
    int num_blocks = num_bytes / blk_size;
    const at::cuda::OptionalCUDAGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    cudaMemcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes, cudaMemcpyHostToDevice);
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

    switch (target_dtype) {
        case torch::kFloat16:
            dequantize_q2_k_fp16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (__half*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kBFloat16:
            dequantize_q2_k_bf16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (nv_bfloat16*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kFloat32:
            dequantize_q2_k_fp32_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), output.data_ptr<float>(), blk_size, ele_per_blk, num_blocks);
            break;
        default:
            printf("target type not support\n");
            exit(0);
    }
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor dequantize_iq4_xs(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype) {
    int num_blocks = num_bytes / blk_size;
    const at::cuda::OptionalCUDAGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    cudaMemcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes, cudaMemcpyHostToDevice);
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

    switch (target_dtype) {
        case torch::kFloat16:
            dequantize_iq4_xs_fp16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (__half*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kBFloat16:
            dequantize_iq4_xs_bf16_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), (nv_bfloat16*)output.data_ptr(), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kFloat32:
            dequantize_iq4_xs_fp32_kernel<<<512, 256>>>(data_gpu.data_ptr<int8_t>(), output.data_ptr<float>(), blk_size, ele_per_blk, num_blocks);
            break;
        default:
            printf("target type not support\n");
            exit(0);
    }
    cudaDeviceSynchronize();
    return output;
}
