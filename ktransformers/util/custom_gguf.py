#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Azure-Tang, Boxin Zhang, chenht2022
Date         : 2024-07-26 08:48:54
Version      : 1.0.0
LastEditors  : kkk1nak0
LastEditTime : 2024-08-14 08:20:45
Adapted from https://github.com/99991/pygguf/blob/main/gguf.py
Copyright (c) 2023-2024 The ggml authors
Copyright (c) 2024 Thomas Germer
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
# copied from llama.cpp/gguf-py/gguf/constants.py to satisfy dependence of gguf
# GGUF specification
# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
import struct
import warnings
import numpy as np
import re
import numpy.typing as npt
from typing import Sequence
import os
from enum import IntEnum
import torch
if not torch.xpu.is_available():
    import KTransformersOps
import ctypes
import math

class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30

QK_K = 256
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32:     (1, 4),
    GGMLQuantizationType.F16:     (1, 2),
    GGMLQuantizationType.Q4_0:    (32, 2 + 16),
    GGMLQuantizationType.Q4_1:    (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0:    (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1:    (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0:    (32, 2 + 32),
    GGMLQuantizationType.Q8_1:    (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K:    (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K:    (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K:    (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K:    (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K:    (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K:    (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS:  (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S:   (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL:  (32, 2 + 16),
    GGMLQuantizationType.IQ3_S:   (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S:   (256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS:  (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.I8:      (1, 1),
    GGMLQuantizationType.I16:     (1, 2),
    GGMLQuantizationType.I32:     (1, 4),
    GGMLQuantizationType.I64:     (1, 8),
    GGMLQuantizationType.F64:     (1, 8),
    GGMLQuantizationType.IQ1_M:   (256, QK_K // 8 + QK_K // 16  + QK_K // 32),
    GGMLQuantizationType.BF16:    (1, 2),
}

# copied from llama.cpp/gguf-py/gguf/quants.py to avoid dependence of gguf
def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})")
    return (*shape[:-1], shape[-1] // block_size * type_size)

GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q5_0": 6,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "IQ3_S": 21,
    "IQ4_XS": 23,
    "BF16": 30,
}

GGML_NAMES = {ggml_type: name for name, ggml_type in GGML_TYPES.items()}

GGML_BLOCK_SIZES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "Q4_0": 2 + 16,
    "Q5_0": 2 + 4 + 16,
    "Q8_0": 2 + 32,
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,
    "Q3_K": 256 // 8 + 256 // 4 + 12 + 2,
    "Q4_K": 2 + 2 + 12 + 256 // 2,
    "Q5_K": 2 + 2 + 12 + 256 // 8 + 256 // 2,
    "Q6_K": 256 // 2 + 256 // 4 + 256 // 16 + 2,
    "IQ3_S": 2 + 256 // 4 + 256 // 32 + 256 // 8 + 4,
    "IQ4_XS": 2 + 2 + 256 // 2 + 256 // 64,
    "FP8": 1,
}

GGML_ELEMENTS_PER_BLOCK = {
    "F32": 1,
    "F16": 1,
    "BF16": 1,
    "Q4_0": 32,
    "Q5_0": 32,
    "Q8_0": 32,
    "Q2_K": 256,
    "Q3_K": 256,
    "Q4_K": 256,
    "Q5_K": 256,
    "Q6_K": 256,
    "IQ3_S": 256,
    "IQ4_XS": 256,
    "FP8": 1,
}

DATA_TYPES = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
    "int64": 11,
    "float64": 12,
    "FP8": 13,
}


def read_value(f, data_type):
    if data_type == DATA_TYPES["string"]:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    elif data_type == DATA_TYPES["bool"]:
        return bool(struct.unpack("<?", f.read(1))[0])

    elif data_type == DATA_TYPES["uint8"]:
        return struct.unpack("<B", f.read(1))[0]

    elif data_type == DATA_TYPES["int8"]:
        return struct.unpack("<b", f.read(1))[0]

    elif data_type == DATA_TYPES["uint16"]:
        return struct.unpack("<H", f.read(2))[0]

    elif data_type == DATA_TYPES["int16"]:
        return struct.unpack("<h", f.read(2))[0]

    elif data_type == DATA_TYPES["uint32"]:
        return struct.unpack("<I", f.read(4))[0]

    elif data_type == DATA_TYPES["int32"]:
        return struct.unpack("<i", f.read(4))[0]

    elif data_type == DATA_TYPES["float32"]:
        return struct.unpack("<f", f.read(4))[0]

    elif data_type == DATA_TYPES["uint64"]:
        return struct.unpack("<Q", f.read(8))[0]

    elif data_type == DATA_TYPES["int64"]:
        return struct.unpack("<q", f.read(8))[0]

    elif data_type == DATA_TYPES["float64"]:
        return struct.unpack("<d", f.read(8))[0]

    elif data_type == DATA_TYPES["array"]:
        elem_type, count = struct.unpack("<IQ", f.read(4 + 8))
        return [read_value(f, elem_type) for _ in range(count)]

    elif data_type == DATA_TYPES["FP8"]:
        return struct.unpack("<B", f.read(1))[0]

    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")

def dequantize_q2_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1547
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L74
    block_size = GGML_BLOCK_SIZES["Q2_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    dmin = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    d = data_f16[:, -2].reshape(num_blocks, 1, 1).astype(np.float32)
    scales = data_u8[:, :16].reshape(num_blocks, 16, 1)
    qs = data_u8[:, 16:80].reshape(num_blocks, 64)

    tmp = np.stack([
        qs[:, 00:16] >> 0,
        qs[:, 16:32] >> 0,
        qs[:, 00:16] >> 2,
        qs[:, 16:32] >> 2,
        qs[:, 00:16] >> 4,
        qs[:, 16:32] >> 4,
        qs[:, 00:16] >> 6,
        qs[:, 16:32] >> 6,
        qs[:, 32:48] >> 0,
        qs[:, 48:64] >> 0,
        qs[:, 32:48] >> 2,
        qs[:, 48:64] >> 2,
        qs[:, 32:48] >> 4,
        qs[:, 48:64] >> 4,
        qs[:, 32:48] >> 6,
        qs[:, 48:64] >> 6,
    ], axis=1)

    return d * (scales & 15) * (tmp & 3) - dmin * (scales >> 4)

def dequantize_q2_k_gpu(data, device:str ="cuda", target_dtype = torch.get_default_dtype()):
    block_size = GGML_BLOCK_SIZES["Q2_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q2_K"]
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    # TODO: this and from_numpy in other functions will cause a warning saying that numpy is not writable, 
    # the best way to fix this is transfer ptr to KTransformersOps instead of Tensor.
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q2_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q3_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1723C32-L1723C42
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L95
    block_size = GGML_BLOCK_SIZES["Q3_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    d = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    bits = np.unpackbits(data_u8[:, :32].reshape(num_blocks, 32, 1), axis=-1, bitorder="little")
    bits = 4 ^ (bits << 2)
    qs = data_u8[:, 32:32 + 64].astype(np.int16)
    a, b, c = data_u8[:, 96: 96 + 12].reshape(num_blocks, 3, 4).transpose(1, 0, 2)
    scales = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
    scales[:, 0] = (a & 15) | ((c & 3) << 4)
    scales[:, 1] = (b & 15) | (((c >> 2) & 3) << 4)
    scales[:, 2] = (a >> 4) | (((c >> 4) & 3) << 4)
    scales[:, 3] = (b >> 4) | ((c >> 6) << 4)
    scales = scales.reshape(num_blocks, 16, 1).astype(np.int16)

    return d * (scales - 32) * np.stack([
        (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0]),
        (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0]),
        (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1]),
        (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1]),
        (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2]),
        (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2]),
        (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3]),
        (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3]),
        (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4]),
        (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4]),
        (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5]),
        (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5]),
        (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6]),
        (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6]),
        (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7]),
        (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7])
    ], axis=1)

def dequantize_q3_k_gpu(data, device:str ="cuda", target_dtype = torch.get_default_dtype()):
    block_size = GGML_BLOCK_SIZES["Q3_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q3_K"]
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    # TODO: this and from_numpy in other functions will cause a warning saying that numpy is not writable, 
    # the best way to fix this is transfer ptr to KTransformersOps instead of Tensor.
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q3_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q4_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    num_blocks = len(data) // block_size
    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)
    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    factors = scale_factors * np.concatenate([qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1)
    offsets = scale_offsets * np.concatenate([qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1)
    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xf, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)
    # Dequantize final weights using scales and offsets
    return factors * qs2 - offsets

def dequantize_q4_k_gpu(data, device:str ="cuda", target_dtype = torch.get_default_dtype()):
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q4_K"]
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    # TODO: this and from_numpy in other functions will cause a warning saying that numpy is not writable, 
    # the best way to fix this is transfer ptr to KTransformersOps instead of Tensor.
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q4_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q5_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2129
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L138
    block_size = GGML_BLOCK_SIZES["Q5_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    d = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    dmin = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    scales = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qh = data_u8[:, 16: 16 + 32].reshape(num_blocks, 32, 1)
    qs = data_u8[:, 48: 48 + 128].reshape(num_blocks, 4, 32)

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    qs_hi_4 = qs >> 4
    qs_lo_4 = qs & 15

    scales_lo_6 = scales[:, :8] & 63
    scales_hi_6 = scales[:, :8] >> 6
    scales_lo_4 = scales[:, 8:] & 15
    scales_hi_4 = scales[:, 8:] >> 4

    m1 = dmin * scales_lo_6[:, 4]
    m2 = dmin * scales_lo_6[:, 5]
    m3 = dmin * scales_lo_6[:, 6]
    m4 = dmin * scales_lo_6[:, 7]
    m5 = dmin * (scales_hi_4[:, 0] | (scales_hi_6[:, 4] << 4))
    m6 = dmin * (scales_hi_4[:, 1] | (scales_hi_6[:, 5] << 4))
    m7 = dmin * (scales_hi_4[:, 2] | (scales_hi_6[:, 6] << 4))
    m8 = dmin * (scales_hi_4[:, 3] | (scales_hi_6[:, 7] << 4))

    d1 = d * scales_lo_6[:, 0]
    d2 = d * scales_lo_6[:, 1]
    d3 = d * scales_lo_6[:, 2]
    d4 = d * scales_lo_6[:, 3]
    d5 = d * (scales_lo_4[:, 0] | (scales_hi_6[:, 0] << 4))
    d6 = d * (scales_lo_4[:, 1] | (scales_hi_6[:, 1] << 4))
    d7 = d * (scales_lo_4[:, 2] | (scales_hi_6[:, 2] << 4))
    d8 = d * (scales_lo_4[:, 3] | (scales_hi_6[:, 3] << 4))

    return np.concatenate([
        d1 * (qs_lo_4[:, 0] + (bits[:, :, 0] << 4)) - m1,
        d2 * (qs_hi_4[:, 0] + (bits[:, :, 1] << 4)) - m2,
        d3 * (qs_lo_4[:, 1] + (bits[:, :, 2] << 4)) - m3,
        d4 * (qs_hi_4[:, 1] + (bits[:, :, 3] << 4)) - m4,
        d5 * (qs_lo_4[:, 2] + (bits[:, :, 4] << 4)) - m5,
        d6 * (qs_hi_4[:, 2] + (bits[:, :, 5] << 4)) - m6,
        d7 * (qs_lo_4[:, 3] + (bits[:, :, 6] << 4)) - m7,
        d8 * (qs_hi_4[:, 3] + (bits[:, :, 7] << 4)) - m8,
    ], axis=1)

def dequantize_q5_k_gpu(data, device:str ="cuda", target_dtype = torch.get_default_dtype()):
    block_size = GGML_BLOCK_SIZES["Q5_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q5_K"]
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    # TODO: this and from_numpy in other functions will cause a warning saying that numpy is not writable, 
    # the best way to fix this is transfer ptr to KTransformersOps instead of Tensor.
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q5_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q6_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, block_size)

    scales = data_f16[:, -1].reshape(num_blocks, 1).astype(np.float32)
    # TODO use uint8 and cast later?
    ql = data_u8[:, :128].astype(np.int16)
    qh = data_u8[:, 128:192].astype(np.int16)
    sc = data_i8[:, 192:208, np.newaxis].astype(np.float32)

    # Unpack bits, subtraction requires signed data type
    q1 = (ql[:,   :32 ] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
    q2 = (ql[:, 32:64 ] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
    q3 = (ql[:,   :32 ] >>  4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
    q4 = (ql[:, 32:64 ] >>  4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
    q5 = (ql[:, 64:96 ] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
    q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
    q7 = (ql[:, 64:96 ] >>  4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
    q8 = (ql[:, 96:128] >>  4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

    # Dequantize
    return scales * np.concatenate([
        sc[:,  0] * q1[:, :16],
        sc[:,  1] * q1[:, 16:],
        sc[:,  2] * q2[:, :16],
        sc[:,  3] * q2[:, 16:],
        sc[:,  4] * q3[:, :16],
        sc[:,  5] * q3[:, 16:],
        sc[:,  6] * q4[:, :16],
        sc[:,  7] * q4[:, 16:],
        sc[:,  8] * q5[:, :16],
        sc[:,  9] * q5[:, 16:],
        sc[:, 10] * q6[:, :16],
        sc[:, 11] * q6[:, 16:],
        sc[:, 12] * q7[:, :16],
        sc[:, 13] * q7[:, 16:],
        sc[:, 14] * q8[:, :16],
        sc[:, 15] * q8[:, 16:],
    ], axis=1) 

# @torch.jit.script
def dequantize_q6_k_gpu(data: np.ndarray, device:str = "cuda", target_dtype = torch.get_default_dtype()):
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q6_K"]
    device = torch.device(device)
    num_blocks = len(data) // block_size
    data = np.frombuffer(data, dtype=data.dtype)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q6_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

kvalues_iq4nl = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)

iq3s_grid = np.array([
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
], dtype=np.uint32)

kmask_iq2xs = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

def dequantize_iq3_s(data):
    # Implementation based on llama.cpp dequantize_row_iq3_s
    # https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
    block_size = GGML_BLOCK_SIZES["IQ3_S"]
    num_blocks = len(data) // block_size
    
    # Parse data according to block_iq3_s structure
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    
    # Extract fields from block_iq3_s structure:
    # ggml_half d (2 bytes)
    d = np.frombuffer(data, dtype=np.float16)[0::block_size//2].astype(np.float32).reshape(num_blocks, 1)
    
    # uint8_t qs[QK_K/4] (64 bytes at offset 2)
    qs = data_u8[:, 2:2+64]
    
    # uint8_t qh[QK_K/32] (8 bytes at offset 66)
    qh = data_u8[:, 66:66+8]
    
    # uint8_t signs[QK_K/8] (32 bytes at offset 74)
    signs = data_u8[:, 74:74+32]
    
    # uint8_t scales[IQ3S_N_SCALE] (4 bytes at offset 106)
    scales = data_u8[:, 106:106+4]
    
    y = np.zeros((num_blocks, 256), dtype=np.float32)
    
    for i in range(num_blocks):
        y_offset = 0
        
        for ib32 in range(0, 8, 2):  # QK_K/32 = 8, step by 2
            # Extract scale factors from 4-bit nibbles
            db1 = d[i, 0] * (1 + 2 * (scales[i, ib32//2] & 0xf))
            db2 = d[i, 0] * (1 + 2 * (scales[i, ib32//2] >> 4))
            
            # Process first 32 elements with db1
            qs_offset = ib32 * 8
            signs_offset = ib32 * 4
            qh_idx = ib32
            
            for l in range(4):
                # Construct grid indices using 8 bits from qs + 1 high bit from qh
                idx1 = qs[i, qs_offset + 2*l + 0] | ((qh[i, qh_idx] << (8-2*l)) & 256)
                idx2 = qs[i, qs_offset + 2*l + 1] | ((qh[i, qh_idx] << (7-2*l)) & 256)
                
                # Get 4-element vectors from grid
                grid1 = np.frombuffer(iq3s_grid[idx1].tobytes(), dtype=np.uint8)
                grid2 = np.frombuffer(iq3s_grid[idx2].tobytes(), dtype=np.uint8)
                
                # Apply signs and scaling
                sign_byte = signs[i, signs_offset + l]
                for j in range(4):
                    sign1 = -1.0 if sign_byte & kmask_iq2xs[j] else 1.0
                    sign2 = -1.0 if sign_byte & kmask_iq2xs[j+4] else 1.0
                    
                    y[i, y_offset + j] = db1 * grid1[j] * sign1
                    y[i, y_offset + j + 4] = db1 * grid2[j] * sign2
                
                y_offset += 8
            
            # Process second 32 elements with db2
            qs_offset += 8
            signs_offset += 4
            qh_idx += 1
            
            for l in range(4):
                # Construct grid indices
                idx1 = qs[i, qs_offset + 2*l + 0] | ((qh[i, qh_idx] << (8-2*l)) & 256)
                idx2 = qs[i, qs_offset + 2*l + 1] | ((qh[i, qh_idx] << (7-2*l)) & 256)
                
                # Get 4-element vectors from grid
                grid1 = np.frombuffer(iq3s_grid[idx1].tobytes(), dtype=np.uint8)
                grid2 = np.frombuffer(iq3s_grid[idx2].tobytes(), dtype=np.uint8)
                
                # Apply signs and scaling
                sign_byte = signs[i, signs_offset + l]
                for j in range(4):
                    sign1 = -1.0 if sign_byte & kmask_iq2xs[j] else 1.0
                    sign2 = -1.0 if sign_byte & kmask_iq2xs[j+4] else 1.0
                    
                    y[i, y_offset + j] = db2 * grid1[j] * sign1
                    y[i, y_offset + j + 4] = db2 * grid2[j] * sign2
                
                y_offset += 8
    
    return y.flatten()

def dequantize_iq3_s_gpu(data: np.ndarray, device: str = "cuda", target_dtype = torch.get_default_dtype()):
    block_size = GGML_BLOCK_SIZES["IQ3_S"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["IQ3_S"]
    device = torch.device(device)
    num_blocks = len(data) // block_size
    data = np.frombuffer(data, dtype=data.dtype)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_iq3_s(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_iq4_xs(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/21d3a308fcb7f31cb9beceaeebad4fb622f3c337/src/ggml-quants.c#L3568
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/21d3a308fcb7f31cb9beceaeebad4fb622f3c337/src/ggml-common.h#L393
    block_size = GGML_BLOCK_SIZES["IQ4_XS"]
    num_blocks = len(data) // block_size

    d = np.frombuffer(data, dtype=np.float16)[0::block_size//2].astype(np.float32).reshape(num_blocks, 1)
    scales_h = np.frombuffer(data, dtype=np.uint16)[1::block_size//2].reshape(num_blocks, 1)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)[:, 4:]
    scales_l = data_u8[:, :4].reshape(num_blocks, 4)
    qs = data_u8[:, 4:].reshape(num_blocks, block_size - 8)

    ls = np.zeros((num_blocks, QK_K // 32), dtype=np.int8)
    for ib in range(QK_K // 32):
        ls[:, ib] = ((scales_l[:, ib // 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h[:, 0] >> 2 * ib) & 3) << 4)

    dl = (d * (ls - 32)).reshape(num_blocks, -1, 1)

    qs_lo_4 = qs[:, :QK_K // 2].reshape(num_blocks, -1, 16) & 0xf
    qs_hi_4 = qs[:, :QK_K // 2].reshape(num_blocks, -1, 16) >> 4

    y = np.zeros((num_blocks, QK_K), dtype=np.float32)
    for ib in range(QK_K // 32):
        y[:, ib*32:(ib*32)+16] = dl[:, ib] * kvalues_iq4nl[qs_lo_4[:, ib]]
        y[:, (ib*32)+16:(ib*32)+32] = dl[:, ib] * kvalues_iq4nl[qs_hi_4[:, ib]]

    return y.flatten()

def dequantize_iq4_xs_gpu(data: np.ndarray, device:str = "cuda", target_dtype = torch.get_default_dtype()):
    block_size = GGML_BLOCK_SIZES["IQ4_XS"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["IQ4_XS"]
    device = torch.device(device)
    num_blocks = len(data) // block_size
    data = np.frombuffer(data, dtype=data.dtype)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_iq4_xs(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q4_0(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1515
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-common.h#L141
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q4_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 8)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 2 + 16)[:, 2:]

    return np.concatenate([
        scales * ((qs & 0xf).astype(np.int8) - 8),
        scales * ((qs >> 4).astype(np.int8) - 8),
    ], axis=1)

def dequantize_q4_0_gpu(data, device:str = "cuda", target_dtype = torch.get_default_dtype()):
    raise NotImplementedError()

def dequantize_q5_0(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1556
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-common.h#L161
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q5_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 2 + 8)[:, :1].astype(np.float32)
    qh = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 2 + 4 + 16)[:, 2:2 + 4]
    qs = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 2 + 4 + 16)[:, 2 + 4:]

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    x0 = ((qs & 0xf).astype(np.int8) | (bits[:, :16] << 4)) - 16
    x1 = ((qs >> 4).astype(np.int8) | (bits[:, 16:] << 4)) - 16

    return np.concatenate([
        scales * x0,
        scales * x1,
    ], axis=1)

def dequantize_q5_0_gpu(data, device:str = "cuda", target_dtype = torch.get_default_dtype()):
    raise NotImplementedError()

def dequantize_q8_0(data):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q8_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]
    return scales * qs

def dequantize_q8_0_gpu(data, device:str = "cuda", target_dtype = torch.get_default_dtype()):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    
    block_size = GGML_BLOCK_SIZES["Q8_0"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q8_0"]
    device = torch.device(device)
    data = np.frombuffer(data, dtype=data.dtype)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q8_0(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)


def dequantize_f32(data):
    return np.frombuffer(data, dtype=np.float32)

def dequantize_f32_gpu(data, device, target_dtype = torch.get_default_dtype()):
    data = np.frombuffer(data, dtype=np.float32)
    res = torch.from_numpy(data.copy())
    res_gpu = torch.empty_like(res, device=device, dtype=target_dtype)
    res_gpu.copy_(res)
    return res_gpu

def dequantize_f16(data):
    return np.frombuffer(data, dtype=np.float16)

def dequantize_f16_gpu(data, device, target_dtype = torch.get_default_dtype()):
    data = np.frombuffer(data, dtype=np.float16)
    res = torch.from_numpy(data.copy())
    res_gpu = torch.empty_like(res, device=device, dtype=target_dtype)
    res_gpu.copy_(res)
    return res_gpu

def dequantize_bf16_gpu(data, device, target_dtype = torch.get_default_dtype()):
    data = np.frombuffer(data, dtype=np.float16)
    res = torch.from_numpy(data.copy())
    res_gpu = torch.empty_like(res, device=device)
    res_gpu.copy_(res)
    return res_gpu

GGML_DEQUANTIZE = {
    "F32": dequantize_f32,
    "F16": dequantize_f16,
    "BF16": dequantize_f16,
    "Q4_0": dequantize_q4_0,
    "Q5_0": dequantize_q5_0,
    "Q8_0": dequantize_q8_0,
    "Q2_K": dequantize_q2_k,
    "Q3_K": dequantize_q3_k,
    "Q4_K": dequantize_q4_k,
    "Q5_K": dequantize_q5_k,
    "Q6_K": dequantize_q6_k,
    "IQ3_S": dequantize_iq3_s,
    "IQ4_XS": dequantize_iq4_xs,
}

GGML_DEQUANTIZE_GPU = {
    "F32": dequantize_f32_gpu,
    "F16": dequantize_f16_gpu,
    "BF16": dequantize_bf16_gpu,
    "Q4_0": dequantize_q4_0_gpu,
    "Q5_0": dequantize_q5_0_gpu,
    "Q8_0": dequantize_q8_0_gpu,
    "Q2_K": dequantize_q2_k_gpu,
    "Q3_K": dequantize_q3_k_gpu,
    "Q4_K": dequantize_q4_k_gpu,
    "Q5_K": dequantize_q5_k_gpu,
    "Q6_K": dequantize_q6_k_gpu,
    "IQ3_S": dequantize_iq3_s_gpu,
    "IQ4_XS": dequantize_iq4_xs_gpu,
}


def translate_name_to_gguf_mixtral(name):
    
    replacement_template = {
        "w1.weight": "ffn_gate",
        "w2.weight": "ffn_down",
        "w3.weight": "ffn_up"
    }  

    pattern = re.compile(r"model.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w\d\.weight)")

    def replace_match(match):
        blk_id = match.group(1)
        expert_id = match.group(2)
        weight_type = match.group(3)
        if weight_type in replacement_template:
            return f"blk.{blk_id}.{replacement_template[weight_type]}.{expert_id}.weight"
        else:
            return match.group(0)

    new_name = re.sub(pattern, replace_match, name)
    
    return new_name

def translate_name_to_gguf(name):

    name = translate_name_to_gguf_mixtral(name)

    name = name.replace("lm_head.", "output.")
    name = name.replace("model.embed_tokens.", "token_embd.")
    name = name.replace("model.norm.", "output_norm.")
    
    name = name.replace("model.layers.", "blk.")
    name = name.replace(".input_layernorm", ".attn_norm")
    name = name.replace(".mlp.down_proj", ".ffn_down")
    name = name.replace(".mlp.gate_proj", ".ffn_gate")
    name = name.replace(".mlp.up_proj", ".ffn_up")
    name = name.replace(".post_attention_layernorm", ".ffn_norm")
    name = name.replace(".self_attn.q_proj", ".attn_q")
    name = name.replace(".self_attn.k_proj", ".attn_k")
    name = name.replace(".self_attn.v_proj", ".attn_v")
    name = name.replace(".self_attn.o_proj", ".attn_output")
    name = name.replace(".self_attn.qkv_proj", ".attn_qkv")
    name = name.replace(".self_attn.kv_a_proj_with_mqa", ".attn_kv_a_mqa")
    name = name.replace(".self_attn.kv_a_layernorm", ".attn_kv_a_norm")
    name = name.replace(".self_attn.kv_b_proj", ".attn_kv_b")
    name = name.replace(".self_attn.q_a_proj", ".attn_q_a")
    name = name.replace(".self_attn.q_a_layernorm", ".attn_q_a_norm")
    name = name.replace(".self_attn.q_b_proj", ".attn_q_b")

    name = name.replace(".self_attn.q_norm", ".attn_q_norm")
    name = name.replace(".self_attn.k_norm", ".attn_k_norm")
    
    name = name.replace(".shared_expert.", ".shared_experts.")
    name = name.replace(".shared_expert_", ".shared_experts_")
    name = name.replace(".gate_up_proj.", ".up_proj")
    
    name = name.replace(".mlp.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".mlp.gate.e_score_correction_bias", ".exp_probs_b.bias")
    name = name.replace(".mlp.gate", ".ffn_gate_inp")
    name = name.replace(".mlp.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".mlp.shared_experts.up_proj", ".ffn_up_shexp")
    name = name.replace(".mlp.shared_experts_gate", ".ffn_gate_inp_shexp")


    name = name.replace(".mlp.experts", "")

    
    name = name.replace(".block_sparse_moe.gate.", ".ffn_gate_inp.")
    name = name.replace(".block_sparse_moe.experts", "")
    
    name = name.replace(".feed_forward.experts", "")
    name = name.replace(".feed_forward.router", ".ffn_gate_inp")
    name = name.replace(".feed_forward.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".feed_forward.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".feed_forward.shared_experts.up_proj", ".ffn_up_shexp")

    return name


if __name__ == '__main__':
    gguf_path = '/mnt/data/model/DeepSeek-Coder-V2-GGUF-WJH'
    loader = GGUFLoader(gguf_path)
    loader.load_gguf_tensor('token_embd.weight')

