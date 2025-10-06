/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/**
 * @brief Load data from a source array into a row-major layout tile.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param idx[in] The index of the tile to load data from.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();

    int row_offset = laneid%(dst.base_tile_rows), col_offset = dst.elements_per_base_tile*(laneid/dst.base_tile_rows);

    uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = dst.base_tile_rows*i + row_offset;
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = dst.base_tile_cols*j + col_offset;
            U2* tmp;
            if constexpr (std::is_same_v<U2, bf16_2>) {
                float4 loaded = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                    std::bit_cast<i32x4>(br),
                    (row*row_stride + col) * sizeof(U),
                    0,
                    0
                ));
                tmp = reinterpret_cast<U2*>(&loaded);
            }
            else { // float2
                float4 loaded[2];
                loaded[0] = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                    std::bit_cast<i32x4>(br),
                    (row*row_stride + col) * sizeof(U),
                    0,
                    0
                ));
                loaded[1] = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                    std::bit_cast<i32x4>(br),
                    (row*row_stride + col + 4) * sizeof(U),
                    0,
                    0
                ));
                tmp = reinterpret_cast<U2*>(loaded);
            }
            #pragma unroll
            for(int k = 0; k < dst.packed_per_thread; k++) {
                dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(tmp[k]);
            }
        }
    }
}

/**
 * @brief Load data from a source array into a column-major layout tile.
 *
 * @tparam RT The column-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    
    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();

    const int row_offset = dst.elements_per_base_tile * (laneid / dst.base_tile_cols);
    const int col_offset = laneid % dst.base_tile_cols;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = i*dst.base_tile_rows + row_offset;
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.base_tile_cols + col_offset;
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                dst.tiles[i][j].data[k].x = base_types::convertor<T, U>::convert(src_ptr[(row+k * 2)*row_stride + col]);
                dst.tiles[i][j].data[k].y = base_types::convertor<T, U>::convert(src_ptr[(row+k * 2 + 1)*row_stride + col]);
            }
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    load<2, RT, GL>(dst, src, idx);
}

/**
 * @brief Store data from a register tile to a destination array in global memory with a row-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    int row_offset = laneid%(src.base_tile_rows), col_offset = src.elements_per_base_tile*(laneid/src.base_tile_rows);

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = src.base_tile_rows*i + row_offset;
        
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = src.base_tile_cols*j + col_offset;
            U2 tmp[src.packed_per_thread];
            #pragma unroll
            for(int k = 0; k < src.packed_per_thread; k++) {
                tmp[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
            }
            if constexpr (std::is_same_v<U2, bf16_2>) { // bf16_2
                *(bytes_16*)&dst_ptr[row*row_stride + col] = *(bytes_16*)tmp;
            }
            else { // float2
                *(bytes_16*)&dst_ptr[row*row_stride + col] = *(bytes_16*)tmp;
                *(bytes_16*)&dst_ptr[row*row_stride + col + 4] = *(bytes_16*)&tmp[2];
            }
        }
    }
}


/**
 * @brief Store data from a register tile to a destination array in global memory with a column-major layout.
 *
 * @tparam RT The register tile type with a column-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();

    const int row_offset = 4*(laneid/src.base_tile_cols), col_offset = laneid%src.base_tile_cols;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        const int row = i*src.base_tile_rows + row_offset;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int col = j*src.base_tile_cols + col_offset;
            dst_ptr[(row+0)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            dst_ptr[(row+1)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            dst_ptr[(row+2)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
            dst_ptr[(row+3)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
            dst_ptr[(row+16)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
            dst_ptr[(row+17)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
            dst_ptr[(row+18)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
            dst_ptr[(row+19)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

}