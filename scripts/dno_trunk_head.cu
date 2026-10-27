// Fused hero trunk. Head: scaled K=7 projection, GELU, 160-to-8 context projection and the nested
// surface products, written as the batched context-FFT input. Tail: context expansion and residual,
// 160x160 mix, GELU lift and the split-K=2 head into branch-major weights. Both reproduce the accepted
// XLA arithmetic: cvt.rna TF32 operands into mma.sync m16n8k8 in increasing k, XLA's GELU/tanh PTX and
// separately rounded FP32 operations.
#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kRows = 16;

// XLA's Triton GEMMs round each operand with cvt.rna before the tensor-core MMA.
__device__ __forceinline__ uint32_t tf32(float x) {
    uint32_t y;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void mma_tf32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// jax.nn.gelu instruction for instruction as XLA emits it, including its fast tanh.
__device__ __forceinline__ float xla_gelu(float x) {
    float y;
    asm("{ .reg .f32 z, c, s, n, d, t; .reg .pred p0, p1;\n\t"
        "mul.rn.f32 z, %1, %1;\n\t"
        "mul.rn.f32 z, %1, z;\n\t"
        "mul.rn.f32 z, z, 0f3D372713;\n\t"
        "add.rn.f32 z, %1, z;\n\t"
        "mul.rn.f32 z, z, 0f3F4C422A;\n\t"
        "setp.ge.f32 p0, z, 0f40FFF644;\n\t"
        "selp.f32 c, 0f40FFF644, z, p0;\n\t"
        "setp.le.f32 p0, c, 0fC0FFF644;\n\t"
        "selp.f32 c, 0fC0FFF644, c, p0;\n\t"
        "abs.f32 s, z;\n\t"
        "setp.lt.f32 p1, s, 0f39D1B717;\n\t"
        "mul.rn.f32 s, c, c;\n\t"
        "fma.rn.f32 n, s, 0fA59F25C0, 0f2A61337E;\n\t"
        "fma.rn.f32 n, s, n, 0fAEBD37FF;\n\t"
        "fma.rn.f32 n, s, n, 0f335C0041;\n\t"
        "fma.rn.f32 n, s, n, 0f3779434A;\n\t"
        "fma.rn.f32 n, s, n, 0f3A270DED;\n\t"
        "fma.rn.f32 n, s, n, 0f3BA059DC;\n\t"
        "mul.rn.f32 n, c, n;\n\t"
        "fma.rn.f32 d, s, 0f35A0D3D8, 0f38F895D6;\n\t"
        "fma.rn.f32 d, s, d, 0f3B14AA05;\n\t"
        "fma.rn.f32 d, s, d, 0f3BA059DD;\n\t"
        "div.full.f32 n, n, d;\n\t"
        "selp.f32 t, c, n, p1;\n\t"
        "add.rn.f32 t, t, 0f3F800000;\n\t"
        "mul.rn.f32 t, t, 0f3F000000;\n\t"
        "mul.rn.f32 %0, %1, t; }"
        : "=f"(y) : "f"(x));
    return y;
}

// XLA's fast F32 tanh, as emitted inside the GELU-lift fusion.
__device__ __forceinline__ float xla_tanh(float x) {
    float y;
    asm("{ .reg .f32 c, s, n, d; .reg .pred p0, p1;\n\t"
        "setp.ge.f32 p0, %1, 0f40FFF644;\n\t"
        "selp.f32 c, 0f40FFF644, %1, p0;\n\t"
        "setp.le.f32 p0, c, 0fC0FFF644;\n\t"
        "selp.f32 c, 0fC0FFF644, c, p0;\n\t"
        "abs.f32 s, %1;\n\t"
        "setp.lt.f32 p1, s, 0f39D1B717;\n\t"
        "mul.rn.f32 s, c, c;\n\t"
        "fma.rn.f32 n, s, 0fA59F25C0, 0f2A61337E;\n\t"
        "fma.rn.f32 n, s, n, 0fAEBD37FF;\n\t"
        "fma.rn.f32 n, s, n, 0f335C0041;\n\t"
        "fma.rn.f32 n, s, n, 0f3779434A;\n\t"
        "fma.rn.f32 n, s, n, 0f3A270DED;\n\t"
        "fma.rn.f32 n, s, n, 0f3BA059DC;\n\t"
        "mul.rn.f32 n, c, n;\n\t"
        "fma.rn.f32 d, s, 0f35A0D3D8, 0f38F895D6;\n\t"
        "fma.rn.f32 d, s, d, 0f3B14AA05;\n\t"
        "fma.rn.f32 d, s, d, 0f3BA059DD;\n\t"
        "div.full.f32 n, n, d;\n\t"
        "selp.f32 %0, c, n, p1; }"
        : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float xla_gelu_lift(float x) {
    float g = xla_gelu(x);
    return __fmul_rn(g, xla_tanh(g));
}

// One block per 16 * MTiles rows (one stage), one warp per 8-column tile of every 16-row tile. Phase 1 forms
// h2 = h1 + (context / N) We, phase 2 the mix GEMM and GELU lift, phase 3 (four warps) the two 80-long head
// half-sums added in FP32. Each warp's mix and head weights sit in registers before their MMA chains; two row
// tiles per block halve the SMs the tail occupies, leaving room for the concurrent G1 kernels.
template <int Width, int Channels, int Branches, int MTiles>
__global__ void __launch_bounds__(4 * Width) trunk_tail_kernel(
        const float* __restrict__ context, const float* __restrict__ h1, const float* __restrict__ w_expand,
        const float* __restrict__ w_mix, const float* __restrict__ w_head, float* __restrict__ weights,
        int points, int64_t total_rows) {
    static_assert(Channels == 8 && Width % 16 == 0 && Branches <= 32, "hero tail shapes");
    constexpr int kSteps = Width / 8, kHalfSteps = kSteps / 2;
    __shared__ float h2[MTiles][kRows][Width + 4];
    __shared__ float g[MTiles][kRows][Width + 4];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gi = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * kRows * MTiles, stage = row0 / points, x0 = row0 % points;
    const int n = 8 * warp + gi, c0 = 8 * warp + 2 * t;

    uint32_t mix_b[kSteps][2];
    for (int s = 0; s < kSteps; ++s) {
        mix_b[s][0] = tf32(w_mix[(8 * s + t) * Width + n]);
        mix_b[s][1] = tf32(w_mix[(8 * s + t + 4) * Width + n]);
    }
    const uint32_t we0 = tf32(w_expand[t * Width + n]), we1 = tf32(w_expand[(t + 4) * Width + n]);
    for (int m = 0; m < MTiles; ++m) {
        const int r0 = kRows * m;
        uint32_t a[4];
        for (int i = 0; i < 4; ++i)
            a[i] = tf32(__fmul_rn(context[(stage * Channels + t + 4 * (i / 2)) * points + x0 + r0 + gi + 8 * (i % 2)],
                                  1.0f / points));
        float skip[4];
        for (int i = 0; i < 4; ++i) skip[i] = h1[(row0 + r0 + gi + 8 * (i / 2)) * Width + c0 + i % 2];
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma_tf32(d, a, we0, we1);
        for (int i = 0; i < 4; ++i) h2[m][gi + 8 * (i / 2)][c0 + i % 2] = __fadd_rn(skip[i], d[i]);
    }
    __syncthreads();

    float mix[MTiles][4] = {};
    for (int s = 0; s < kSteps; ++s) {
        const int k = 8 * s;
        for (int m = 0; m < MTiles; ++m) {
            uint32_t ah[4] = {tf32(h2[m][gi][k + t]), tf32(h2[m][gi + 8][k + t]),
                              tf32(h2[m][gi][k + t + 4]), tf32(h2[m][gi + 8][k + t + 4])};
            mma_tf32(mix[m], ah, mix_b[s][0], mix_b[s][1]);
        }
    }
    for (int m = 0; m < MTiles; ++m)
        for (int i = 0; i < 4; ++i) g[m][gi + 8 * (i / 2)][c0 + i % 2] = xla_gelu_lift(mix[m][i]);
    __syncthreads();

    if (warp >= 4 * MTiles) return;
    const int m = warp / 4, b = 8 * (warp % 4) + gi;
    uint32_t head_b[kSteps][2];
    for (int s = 0; s < kSteps; ++s)
        for (int j = 0; j < 2; ++j)
            head_b[s][j] = tf32(b < Branches ? w_head[(8 * s + t + 4 * j) * Branches + b] : 0.0f);
    float first[4] = {0.0f, 0.0f, 0.0f, 0.0f}, second[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < kHalfSteps; ++s) {
        const int k0 = 8 * s, k1 = 8 * (s + kHalfSteps);
        uint32_t a0[4] = {tf32(g[m][gi][k0 + t]), tf32(g[m][gi + 8][k0 + t]), tf32(g[m][gi][k0 + t + 4]), tf32(g[m][gi + 8][k0 + t + 4])};
        uint32_t a1[4] = {tf32(g[m][gi][k1 + t]), tf32(g[m][gi + 8][k1 + t]), tf32(g[m][gi][k1 + t + 4]), tf32(g[m][gi + 8][k1 + t + 4])};
        mma_tf32(first, a0, head_b[s][0], head_b[s][1]);
        mma_tf32(second, a1, head_b[s + kHalfSteps][0], head_b[s + kHalfSteps][1]);
    }
    for (int i = 0; i < 4; ++i) {
        const int column = 8 * (warp % 4) + 2 * t + i % 2;
        if (column < Branches)
            weights[column * total_rows + row0 + kRows * m + gi + 8 * (i / 2)] = __fadd_rn(first[i], second[i]);
    }
}

ffi::Error trunk_tail(cudaStream_t stream, ffi::AnyBuffer context, ffi::AnyBuffer h1, ffi::AnyBuffer w_expand,
                      ffi::AnyBuffer w_mix, ffi::AnyBuffer w_head, ffi::Result<ffi::AnyBuffer> weights,
                      int64_t m_tiles) {
    const int points = static_cast<int>(context.dimensions().back());
    const int64_t rows = h1.element_count() / 160;
    if (w_mix.element_count() != 160 * 160 || w_expand.element_count() != 8 * 160 || w_head.element_count() != 160 * 28
            || points % (kRows * m_tiles) != 0)
        return ffi::Error::InvalidArgument("Trunk tail supports width 160, 8 context channels and 28 branches.");
    const auto* args = static_cast<const float*>(context.untyped_data());
    if (m_tiles == 2)
        trunk_tail_kernel<160, 8, 28, 2><<<rows / (2 * kRows), 4 * 160, 0, stream>>>(
            args, static_cast<const float*>(h1.untyped_data()), static_cast<const float*>(w_expand.untyped_data()),
            static_cast<const float*>(w_mix.untyped_data()), static_cast<const float*>(w_head.untyped_data()),
            static_cast<float*>(weights->untyped_data()), points, rows);
    else
        trunk_tail_kernel<160, 8, 28, 1><<<rows / kRows, 4 * 160, 0, stream>>>(
            args, static_cast<const float*>(h1.untyped_data()), static_cast<const float*>(w_expand.untyped_data()),
            static_cast<const float*>(w_mix.untyped_data()), static_cast<const float*>(w_head.untyped_data()),
            static_cast<float*>(weights->untyped_data()), points, rows);
    auto error = cudaGetLastError();
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

// One block per 16 rows (points of one stage); its warps split the first projection's 8-column tiles,
// then warp 0 forms the context projection while the others store h1 and the nested products.
template <int Width, int Channels, int TilesPerWarp>
__global__ void __launch_bounds__(32 * Width / (8 * TilesPerWarp)) trunk_head_kernel(
        const float* __restrict__ features, const float* __restrict__ reciprocal,
        const float* __restrict__ w_first, const float* __restrict__ w_context,
        const float* __restrict__ normalized, const float* __restrict__ fields,
        float* __restrict__ h1, float* __restrict__ spectrum_input,
        int n_features, int points, int n_fields, int nested_from, float nested_reciprocal) {
    static_assert(Channels == 8 && Width % (8 * TilesPerWarp) == 0, "one n-tile of context channels");
    constexpr int kWarps = Width / (8 * TilesPerWarp);
    __shared__ float h[kRows][Width + 4];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * kRows, stage = row0 / points, x0 = row0 % points;
    const int out_channels = Channels + n_fields - nested_from;

    // A fragment: rows g and g+8, columns t and t+4 of the scaled features.
    uint32_t a[4];
    for (int i = 0; i < 4; ++i) {
        int r = g + 8 * (i % 2), k = t + 4 * (i / 2);
        a[i] = tf32(k < n_features ? __fmul_rn(features[(stage * n_features + k) * points + x0 + r], reciprocal[k]) : 0.0f);
    }
    for (int jj = 0; jj < TilesPerWarp; ++jj) {
        const int n = 8 * (warp * TilesPerWarp + jj) + g;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma_tf32(d, a, tf32(t < n_features ? w_first[t * Width + n] : 0.0f),
                 tf32(t + 4 < n_features ? w_first[(t + 4) * Width + n] : 0.0f));
        const int c = n - g + 2 * t;
        h[g][c] = xla_gelu(d[0]);
        h[g][c + 1] = xla_gelu(d[1]);
        h[g + 8][c] = xla_gelu(d[2]);
        h[g + 8][c + 1] = xla_gelu(d[3]);
    }
    __syncthreads();

    if (warp == 0) {
        float context[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = 0; k < Width; k += 8) {
            uint32_t ah[4] = {tf32(h[g][k + t]), tf32(h[g + 8][k + t]), tf32(h[g][k + t + 4]), tf32(h[g + 8][k + t + 4])};
            mma_tf32(context, ah, tf32(w_context[(k + t) * Channels + g]), tf32(w_context[(k + t + 4) * Channels + g]));
        }
        for (int i = 0; i < 4; ++i) {
            int r = g + 8 * (i / 2), c = 2 * t + i % 2;
            spectrum_input[(stage * out_channels + c) * points + x0 + r] = context[i];
        }
    }
    for (int i = threadIdx.x; i < kRows * Width / 4; i += blockDim.x) {
        int r = i / (Width / 4), c = 4 * (i % (Width / 4));
        *reinterpret_cast<float4*>(h1 + (row0 + r) * Width + c) = make_float4(h[r][c], h[r][c + 1], h[r][c + 2], h[r][c + 3]);
    }
    for (int i = threadIdx.x; i < kRows * (n_fields - nested_from); i += blockDim.x) {
        int r = i % kRows, j = i / kRows;
        float surface = __fmul_rn(normalized[stage * 2 * points + x0 + r], nested_reciprocal);
        spectrum_input[(stage * out_channels + Channels + j) * points + x0 + r] =
            __fmul_rn(surface, fields[(stage * n_fields + nested_from + j) * points + x0 + r]);
    }
}

template <int TilesPerWarp>
void launch_trunk_head(cudaStream_t stream, int64_t rows, const float* features, const float* reciprocal,
                       const float* w_first, const float* w_context, const float* normalized, const float* fields,
                       float* h1, float* spectrum_input, int n_features, int points, int n_fields, int nested_from,
                       float nested_reciprocal) {
    trunk_head_kernel<160, 8, TilesPerWarp><<<rows / kRows, 32 * 160 / (8 * TilesPerWarp), 0, stream>>>(
        features, reciprocal, w_first, w_context, normalized, fields, h1, spectrum_input,
        n_features, points, n_fields, nested_from, nested_reciprocal);
}

ffi::Error trunk_head(cudaStream_t stream, ffi::AnyBuffer features, ffi::AnyBuffer reciprocal,
                      ffi::AnyBuffer w_first, ffi::AnyBuffer w_context, ffi::AnyBuffer normalized,
                      ffi::AnyBuffer fields, ffi::Result<ffi::AnyBuffer> h1,
                      ffi::Result<ffi::AnyBuffer> spectrum_input, float nested_reciprocal, int64_t nested_from,
                      int64_t tiles_per_warp) {
    const auto feature_dims = features.dimensions(), field_dims = fields.dimensions();
    const int points = static_cast<int>(feature_dims.back());
    const int n_features = static_cast<int>(feature_dims[feature_dims.size() - 2]);
    const int n_fields = static_cast<int>(field_dims[field_dims.size() - 2]);
    const int width = static_cast<int>(w_first.dimensions().back());
    const int channels = static_cast<int>(w_context.dimensions().back());
    const int64_t rows = features.element_count() / n_features;
    if (width != 160 || channels != 8 || n_features > 8 || points % kRows != 0)
        return ffi::Error::InvalidArgument("Trunk head supports width 160, 8 context channels and <=8 features.");
    auto launch = tiles_per_warp == 1 ? launch_trunk_head<1> : tiles_per_warp == 2 ? launch_trunk_head<2>
                  : tiles_per_warp == 4 ? launch_trunk_head<4> : launch_trunk_head<5>;
    launch(stream, rows, static_cast<const float*>(features.untyped_data()),
           static_cast<const float*>(reciprocal.untyped_data()), static_cast<const float*>(w_first.untyped_data()),
           static_cast<const float*>(w_context.untyped_data()), static_cast<const float*>(normalized.untyped_data()),
           static_cast<const float*>(fields.untyped_data()), static_cast<float*>(h1->untyped_data()),
           static_cast<float*>(spectrum_input->untyped_data()), n_features, points, n_fields,
           static_cast<int>(nested_from), nested_reciprocal);
    auto error = cudaGetLastError();
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_trunk_head, trunk_head,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Attr<float>("nested_reciprocal").Attr<int64_t>("nested_from").Attr<int64_t>("tiles_per_warp"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_trunk_tail, trunk_tail,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Attr<int64_t>("m_tiles"),
    {ffi::Traits::kCmdBufferCompatible});
