// Fused hero trunk head: scaled K=7 projection, GELU, 160-to-8 context projection and the
// nested surface products, written as the batched context-FFT input. Reproduces the accepted
// XLA arithmetic: cvt.rna TF32 operands into mma.sync m16n8k8 in increasing k, XLA's GELU/tanh PTX and
// separately rounded FP32 multiplies.
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
