// Fused hero trunk. Head: scaled K=7 projection, GELU, 160-to-8 context projection and the nested
// surface products, written as the batched context-FFT input. Tail: context expansion and residual,
// 160x160 mix, GELU lift and the split-K=2 head into branch-major weights. Both reproduce the accepted
// XLA arithmetic: cvt.rna TF32 operands into mma.sync m16n8k8 in increasing k, XLA's GELU/tanh PTX and
// separately rounded FP32 operations.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// The four 8x4 b32 sub-matrices of a 16x8 TF32 A tile in mma fragment order; each lane supplies one row address.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const uint32_t* address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(address))));
}

// Exact TF32 B operand from a power-of-two-scaled FP16 copy.
__device__ __forceinline__ uint32_t tf32_from_half(__half x, float inverse_scale) {
    return __float_as_uint(__half2float(x) * inverse_scale);
}

// One block per 16 * MTiles rows (one stage), one warp per 8-column tile of every 16-row tile. Phase 1 forms
// h2 = h1 + (context / N) We, phase 2 the mix GEMM and GELU lift, phase 3 (one warp per 8-branch tile and row tile)
// the two half-width head sums added in FP32. h2 and g are kept as TF32 bits for ldmatrix; the mix and head weights
// arrive as exact FP16 copies in fragment order (see pack_tail_weights), Width / 32 16-byte chunks per lane, since
// every block streams them from L2.
template <int Width, int Channels, int Branches, int MTiles>
__global__ void __launch_bounds__(4 * Width) trunk_tail_kernel(
        const float* __restrict__ context, const float* __restrict__ h1, const float* __restrict__ w_expand,
        const uint4* __restrict__ mix16, const uint4* __restrict__ head16, float mix_scale, float head_scale,
        float* __restrict__ weights, int points, int64_t total_rows) {
    constexpr int kSteps = Width / 8, kHalfSteps = kSteps / 2, kChunks = kSteps / 4, kHeadTiles = (Branches + 7) / 8;
    static_assert(Channels == 8 && Width % 32 == 0 && kHeadTiles * MTiles <= Width / 8, "tail shapes");
    __shared__ __align__(16) uint32_t h2[MTiles][kRows][Width + 4];
    __shared__ __align__(16) uint32_t g[MTiles][kRows][Width + 4];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gi = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * kRows * MTiles, stage = row0 / points, x0 = row0 % points;
    const int n = 8 * warp + gi, c0 = 8 * warp + 2 * t;
    const int lr = lane % 8 + 8 * (lane / 8 % 2), lc = 4 * (lane / 16);

    uint4 mix_raw[kChunks];
    for (int c = 0; c < kChunks; ++c) mix_raw[c] = mix16[(warp * kChunks + c) * 32 + lane];
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
        for (int i = 0; i < 4; ++i) h2[m][gi + 8 * (i / 2)][c0 + i % 2] = tf32(__fadd_rn(skip[i], d[i]));
    }
    const __half* mix_h = reinterpret_cast<const __half*>(mix_raw);
    __syncthreads();

    float mix[MTiles][4] = {};
    for (int s = 0; s < kSteps; ++s) {
        const uint32_t b0 = tf32_from_half(mix_h[2 * s], mix_scale), b1 = tf32_from_half(mix_h[2 * s + 1], mix_scale);
        for (int m = 0; m < MTiles; ++m) {
            uint32_t ah[4];
            ldmatrix_x4(ah, &h2[m][lr][8 * s + lc]);
            mma_tf32(mix[m], ah, b0, b1);
        }
    }
    // Head warps fetch their weights before the GELU lift so the load latency overlaps it.
    const bool head = warp < kHeadTiles * MTiles;
    const int m = warp / kHeadTiles;
    uint4 head_raw[kChunks];
    if (head)
        for (int c = 0; c < kChunks; ++c) head_raw[c] = head16[(warp % kHeadTiles * kChunks + c) * 32 + lane];
    for (int m = 0; m < MTiles; ++m)
        for (int i = 0; i < 4; ++i) g[m][gi + 8 * (i / 2)][c0 + i % 2] = tf32(xla_gelu_lift(mix[m][i]));
    __syncthreads();

    if (!head) return;
    const __half* head_h = reinterpret_cast<const __half*>(head_raw);
    float first[4] = {0.0f, 0.0f, 0.0f, 0.0f}, second[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < kHalfSteps; ++s) {
        const int s1 = s + kHalfSteps;
        uint32_t a0[4], a1[4];
        ldmatrix_x4(a0, &g[m][lr][8 * s + lc]);
        ldmatrix_x4(a1, &g[m][lr][8 * s1 + lc]);
        mma_tf32(first, a0, tf32_from_half(head_h[2 * s], head_scale), tf32_from_half(head_h[2 * s + 1], head_scale));
        mma_tf32(second, a1, tf32_from_half(head_h[2 * s1], head_scale), tf32_from_half(head_h[2 * s1 + 1], head_scale));
    }
    for (int i = 0; i < 4; ++i) {
        const int column = 8 * (warp % kHeadTiles) + 2 * t + i % 2;
        if (column < Branches)
            weights[column * total_rows + row0 + kRows * m + gi + 8 * (i / 2)] = __fadd_rn(first[i], second[i]);
    }
}

template <int Width, int Branches>
void launch_trunk_tail(cudaStream_t stream, int64_t m_tiles, int64_t rows, const float* context, const float* h1,
                       const float* w_expand, const uint4* mix16, const uint4* head16, float mix_scale,
                       float head_scale, float* weights, int points) {
    // Two row tiles per block only while their h2 and g tiles fit in 48 KB of static shared memory.
    constexpr int kMaxTiles = 2 * 2 * kRows * (Width + 4) * 4 <= 48 * 1024 ? 2 : 1;
    auto run = m_tiles == 2 ? trunk_tail_kernel<Width, 8, Branches, kMaxTiles> : trunk_tail_kernel<Width, 8, Branches, 1>;
    run<<<rows / (m_tiles * kRows), 4 * Width, 0, stream>>>(context, h1, w_expand, mix16, head16, mix_scale, head_scale,
                                                           weights, points, rows);
}

ffi::Error trunk_tail(cudaStream_t stream, ffi::AnyBuffer context, ffi::AnyBuffer h1, ffi::AnyBuffer w_expand,
                      ffi::AnyBuffer mix16, ffi::AnyBuffer head16, ffi::Result<ffi::AnyBuffer> weights,
                      float mix_scale, float head_scale, int64_t m_tiles) {
    const int points = static_cast<int>(context.dimensions().back());
    const int width = static_cast<int>(w_expand.element_count() / 8);
    const int branches = static_cast<int>(weights->dimensions()[0]);
    const int64_t rows = h1.element_count() / width;
    const bool supported = (width == 160 && (branches == 28 || branches == 60))
                           || (width == 256 && branches == 28 && m_tiles == 1);
    if (!supported || mix16.element_count() != int64_t(width) * width
            || head16.element_count() != int64_t(width) * 8 * ((branches + 7) / 8) || points % (kRows * m_tiles) != 0)
        return ffi::Error::InvalidArgument("Trunk tail supports width 160 with 28/60 branches and width 256 with 28.");
    auto run = width == 256 ? launch_trunk_tail<256, 28> : branches == 60 ? launch_trunk_tail<160, 60>
               : launch_trunk_tail<160, 28>;
    run(stream, m_tiles, rows, static_cast<const float*>(context.untyped_data()),
        static_cast<const float*>(h1.untyped_data()), static_cast<const float*>(w_expand.untyped_data()),
        static_cast<const uint4*>(mix16.untyped_data()), static_cast<const uint4*>(head16.untyped_data()),
        mix_scale, head_scale, static_cast<float*>(weights->untyped_data()), points);
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
    __shared__ __align__(16) float h[kRows][Width + 4];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * kRows, stage = row0 / points, x0 = row0 % points;
    const int out_channels = Channels + n_fields - nested_from;
    const int lr = lane % 8 + 8 * (lane / 8 % 2), lc = 4 * (lane / 16);

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
            uint32_t ah[4];
            ldmatrix_x4(ah, reinterpret_cast<const uint32_t*>(&h[lr][k + lc]));
            for (int i = 0; i < 4; ++i) ah[i] = tf32(__uint_as_float(ah[i]));
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

template <int Width, int TilesPerWarp>
void launch_trunk_head(cudaStream_t stream, int64_t rows, const float* features, const float* reciprocal,
                       const float* w_first, const float* w_context, const float* normalized, const float* fields,
                       float* h1, float* spectrum_input, int n_features, int points, int n_fields, int nested_from,
                       float nested_reciprocal) {
    trunk_head_kernel<Width, 8, TilesPerWarp><<<rows / kRows, 32 * Width / (8 * TilesPerWarp), 0, stream>>>(
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
    if ((width != 160 && width != 256) || channels != 8 || n_features > 8 || points % kRows != 0)
        return ffi::Error::InvalidArgument("Trunk head supports widths 160/256, 8 context channels and <=8 features.");
    auto launch = width == 256 ? (tiles_per_warp == 2 ? launch_trunk_head<256, 2> : launch_trunk_head<256, 1>)
                  : tiles_per_warp == 1 ? launch_trunk_head<160, 1> : tiles_per_warp == 2 ? launch_trunk_head<160, 2>
                  : tiles_per_warp == 4 ? launch_trunk_head<160, 4> : launch_trunk_head<160, 5>;
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
        .Ret<ffi::AnyBuffer>().Attr<float>("mix_scale").Attr<float>("head_scale").Attr<int64_t>("m_tiles"),
    {ffi::Traits::kCmdBufferCompatible});
