// Fused hero context filter: one cuFFTDx forward FFT per batched context/nested row, then either the
// real context multiplier and an unnormalized inverse FFT, or the nested inner filter and the existing
// unit-symbol spectral sandwich with surface gamma * eta / s0. Replaces cuFFT forward/inverse calls,
// their multiplies and copies; FFT rounding differs from cuFFT, so this is not bit-identical.
#include <cuda_runtime.h>
#include <cufftdx.hpp>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace cufftdx;

#ifndef DNO_SM
#define DNO_SM 890
#endif

namespace {

template <int N, int EPT>
using Base = decltype(Block() + Size<N>() + Precision<float>() + SM<DNO_SM>() + ElementsPerThread<EPT>() + FFTsPerBlock<1>());
using Folded = RealFFTOptions<complex_layout::natural, real_mode::folded>;

// Rows are stage-major over (context channels, nested channels).
template <class Forward, class Inverse>
__global__ void context_filter_kernel(const float* __restrict__ spectrum_input, const float* __restrict__ context_symbol,
                                      const float* __restrict__ inner_filters, const float* __restrict__ gamma,
                                      const float* __restrict__ normalized, float* __restrict__ context,
                                      float2* __restrict__ nested, int channels, int n_nested, float nested_reciprocal) {
    using C = typename Forward::value_type;
    constexpr int N = size_of<Forward>::value, F = N / 2 + 1;
    static_assert(Forward::output_ept == Inverse::input_ept && Forward::stride == Inverse::stride, "shared register layout");
    extern __shared__ __align__(16) unsigned char scratch[];
    const int rows = channels + n_nested, stage = blockIdx.x / rows, c = blockIdx.x % rows;
    C data[Forward::storage_size] = {};
    for (int i = 0; i < Forward::input_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < Forward::input_length)
            reinterpret_cast<float2*>(data)[i] = reinterpret_cast<const float2*>(spectrum_input)[blockIdx.x * N / 2 + k];
    }
    Forward().execute(data, scratch);
    const bool is_context = c < channels;
    for (int i = 0; i < Inverse::input_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < F) {
            float s = is_context ? context_symbol[k * channels + c] : inner_filters[(c - channels) * F + k];
            data[i] = C(__fmul_rn(data[i].x, s), __fmul_rn(data[i].y, s));
            if (k == 0 || k == N / 2) data[i].y = 0;
        }
    }
    __syncthreads();
    Inverse().execute(data, scratch);
    if (is_context) {
        for (int i = 0; i < Inverse::output_ept; ++i) {
            int k = threadIdx.x + i * Inverse::stride;
            if (k < Inverse::output_length)
                reinterpret_cast<float2*>(context)[(stage * channels + c) * N / 2 + k] = reinterpret_cast<float2*>(data)[i];
        }
        return;
    }
    const int j = c - channels;
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) {
            float2 eta = reinterpret_cast<const float2*>(normalized)[stage * N + k];
            float2 surface = {__fmul_rn(gamma[j], __fmul_rn(eta.x, nested_reciprocal)),
                              __fmul_rn(gamma[j], __fmul_rn(eta.y, nested_reciprocal))};
            data[i] *= float(1.0 / N);
            data[i].x *= surface.x;
            data[i].y *= surface.y;
        }
    }
    __syncthreads();
    Forward().execute(data, scratch);
    for (int i = 0; i < Forward::output_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < F) {
            C value = data[i];
            if (k == 0 || k == N / 2) value.y = 0;
            nested[(stage * n_nested + j) * F + k] = make_float2(value.x, value.y);
        }
    }
}

template <int EPT>
ffi::Error launch(cudaStream_t stream, const float* spectrum_input, const float* context_symbol,
                  const float* inner_filters, const float* gamma, const float* normalized, float* context,
                  float2* nested, int64_t stages, int channels, int n_nested, float nested_reciprocal) {
    using Forward = decltype(Base<1024, EPT>() + Type<fft_type::r2c>() + Folded());
    using Inverse = decltype(Base<1024, EPT>() + Type<fft_type::c2r>() + Folded());
    constexpr int shared = Forward::shared_memory_size > Inverse::shared_memory_size
                           ? Forward::shared_memory_size : Inverse::shared_memory_size;
    context_filter_kernel<Forward, Inverse><<<stages * (channels + n_nested), Forward::block_dim, shared, stream>>>(
        spectrum_input, context_symbol, inner_filters, gamma, normalized, context, nested, channels, n_nested,
        nested_reciprocal);
    auto error = cudaGetLastError();
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

ffi::Error context_filter(cudaStream_t stream, ffi::AnyBuffer spectrum_input, ffi::AnyBuffer context_symbol,
                          ffi::AnyBuffer inner_filters, ffi::AnyBuffer gamma, ffi::AnyBuffer normalized,
                          ffi::Result<ffi::AnyBuffer> context, ffi::Result<ffi::AnyBuffer> nested,
                          float nested_reciprocal, int64_t ept) {
    const auto dims = spectrum_input.dimensions();
    const int points = static_cast<int>(dims.back()), rows = static_cast<int>(dims[dims.size() - 2]);
    const int channels = static_cast<int>(context_symbol.dimensions().back());
    const int n_nested = static_cast<int>(gamma.element_count());
    const int64_t stages = spectrum_input.element_count() / (int64_t(rows) * points);
    if (points != 1024 || rows != channels + n_nested)
        return ffi::Error::InvalidArgument("Context filter supports 1024 points and context+nested rows.");
    auto run = ept == 4 ? launch<4> : ept == 16 ? launch<16> : launch<8>;
    return run(stream, static_cast<const float*>(spectrum_input.untyped_data()),
               static_cast<const float*>(context_symbol.untyped_data()),
               static_cast<const float*>(inner_filters.untyped_data()), static_cast<const float*>(gamma.untyped_data()),
               static_cast<const float*>(normalized.untyped_data()), static_cast<float*>(context->untyped_data()),
               static_cast<float2*>(nested->untyped_data()), stages, channels, n_nested, nested_reciprocal);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_context_filter, context_filter,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Attr<float>("nested_reciprocal").Attr<int64_t>("ept"),
    {ffi::Traits::kCmdBufferCompatible});
