#include "fwht.hpp"

#include <cmath>

template <int N, typename T, bool has_signs>
static void fwht_kernel(const T * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const float * __restrict__ signs, const int n_blk,
                        const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = (float) src[i * WARP_SIZE + lane] * scale;
        if constexpr (has_signs) {
            reg[i] *= signs_row[i * WARP_SIZE + lane];
        }
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

// Wide rows: one work-group per row, butterflies through local memory.
template <int N, typename T, bool has_signs>
static void fwht_kernel_slm(const T * __restrict__ src, float * __restrict__ dst, const float scale,
                            const float * __restrict__ signs, const int n_blk, float * s, const sycl::nd_item<1> & item) {
    const int64_t r   = item.get_group(0);
    const int     tid = item.get_local_id(0);
    const int     nt  = item.get_local_range(0);

    src += r * N;
    dst += r * N;
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    for (int i = tid; i < N; i += nt) {
        float v = (float) src[i] * scale;
        if constexpr (has_signs) {
            v *= signs_row[i];
        }
        s[i] = v;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int h = 1; h < N; h *= 2) {
        for (int idx = tid; idx < N / 2; idx += nt) {
            const int   j = (idx / h) * 2 * h + idx % h;
            const float x = s[j];
            const float y = s[j + h];
            s[j]     = x + y;
            s[j + h] = x - y;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    for (int i = tid; i < N; i += nt) {
        dst[i] = s[i];
    }
}

template <int N, typename T, bool has_signs>
static void launch_fwht(const T * src, float * dst, const int64_t n_rows, const float scale, const float * signs,
                        const int n_blk, dpct::queue_ptr stream) {
    if constexpr (N <= 1024) {
        constexpr int rows_per_block = 4;

        const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

        // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
        const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
        const sycl::range<2> local(rows_per_block, WARP_SIZE);

        stream->parallel_for(sycl::nd_range<2>(global, local),
                             [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 fwht_kernel<N, T, has_signs>(src, dst, n_rows, scale, signs, n_blk, item);
                             });
    } else {
        constexpr int nt = 256;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> s(sycl::range<1>(N), cgh);
            cgh.parallel_for(sycl::nd_range<1>(n_rows * nt, nt), [=](sycl::nd_item<1> item) {
                fwht_kernel_slm<N, T, has_signs>(src, dst, scale, signs, n_blk,
                                                 s.get_multi_ptr<sycl::access::decorated::no>().get(), item);
            });
        });
    }
}

template <typename T>
static bool fwht_dispatch(const T * src_d, float * dst_d, const int n, const int64_t rows, const float scale,
                          const float * signs, const int n_blk, dpct::queue_ptr stream) {
    switch (n) {
#define FWHT_CASE(NN)                                                                      \
        case NN:                                                                           \
            if (signs) {                                                                   \
                launch_fwht<NN, T, true>(src_d, dst_d, rows, scale, signs, n_blk, stream); \
            } else {                                                                       \
                launch_fwht<NN, T, false>(src_d, dst_d, rows, scale, nullptr, 1, stream);  \
            }                                                                              \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        FWHT_CASE(512)
        FWHT_CASE(1024)
        FWHT_CASE(2048)
        FWHT_CASE(4096)
        FWHT_CASE(8192)
#undef FWHT_CASE
        default:
            return false;
    }
}

static bool fwht_run(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                     const ggml_tensor * signs_t) {
    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_nelements(src) != ggml_nelements(dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int     n    = (int) dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    const float * signs = nullptr;
    int           n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = (int) (signs_t->ne[0] / n);
    }

    const float scale = 1.0f / std::sqrt((float) n);
    if (src->type == GGML_TYPE_F32) {
        return fwht_dispatch((const float *) src->data, (float *) dst->data, n, rows, scale, signs, n_blk, ctx.stream());
    }
    return fwht_dispatch((const sycl::half *) src->data, (float *) dst->data, n, rows, scale, signs, n_blk, ctx.stream());
}

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }
    return fwht_run(ctx, src, dst, nullptr);
}

bool ggml_sycl_op_fwht_signed(ggml_backend_sycl_context & ctx, const ggml_tensor * src, const ggml_tensor * signs,
                              ggml_tensor * dst) {
    return fwht_run(ctx, src, dst, signs);
}
