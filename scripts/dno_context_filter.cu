// Fused hero context filter: one cuFFTDx forward FFT per batched context/nested row, then either the
// real context multiplier and an unnormalized inverse FFT, or the nested inner filter and the existing
// unit-symbol spectral sandwich with surface gamma * eta / s0. Replaces cuFFT forward/inverse calls,
// their multiplies and copies; FFT rounding differs from cuFFT, so this is not bit-identical.
// Also hosts the hero's fused FP64 M1 G1 kernels (g1_kernel, bit-identical to the accepted G1, and
// exact_g1_kernel from the FP64 state spectra) and the FP32 trunk-input transforms (trunk_input_kernel,
// trunk_fields_kernel).
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

// Rows are stage-major over (context channels, nested channels); leading index = stage * batch + trajectory,
// and the depth-dependent context symbol / inner filters are indexed by trajectory = leading % n_depth.
template <class Forward, class Inverse>
__global__ void context_filter_kernel(const float* __restrict__ spectrum_input, const float* __restrict__ context_symbol,
                                      const float* __restrict__ inner_filters, const float* __restrict__ gamma,
                                      const float* __restrict__ normalized, float* __restrict__ context,
                                      float2* __restrict__ nested, int channels, int n_nested, int n_depth,
                                      float nested_reciprocal) {
    using C = typename Forward::value_type;
    constexpr int N = size_of<Forward>::value, F = N / 2 + 1;
    static_assert(Forward::output_ept == Inverse::input_ept && Forward::stride == Inverse::stride, "shared register layout");
    extern __shared__ __align__(16) unsigned char scratch[];
    const int rows = channels + n_nested, stage = blockIdx.x / rows, c = blockIdx.x % rows, depth = stage % n_depth;
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
            float s = is_context ? context_symbol[(depth * F + k) * channels + c]
                                 : inner_filters[(depth * n_nested + c - channels) * F + k];
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

template <int N>
using Base64 = decltype(Block() + Size<N>() + Precision<double>() + SM<DNO_SM>() + ElementsPerThread<8>() + FFTsPerBlock<1>());
constexpr int kMaxG1Rows = 4096;
__device__ unsigned g1_arrivals[kMaxG1Rows];

__device__ unsigned g1_ready[kMaxG1Rows];

template <class Forward, class Inverse>
constexpr int g1_shared() {
    return Forward::shared_memory_size > Inverse::shared_memory_size ? Forward::shared_memory_size
                                                                     : Inverse::shared_memory_size;
}

// Shared G1 epilogue: forward transform of the product, operator, and the later of a row's two operator blocks
// writes -(L_0 term + L_1 term).
template <class Forward, class C>
__device__ __forceinline__ void g1_finish(C (&data)[Forward::storage_size], unsigned char* scratch,
                                          const double2* symbol, double2* partial, double2* out, int row, int op) {
    constexpr int N = size_of<Forward>::value, F = N / 2 + 1;
    __shared__ bool last;
    __syncthreads();
    Forward().execute(data, scratch);
    for (int i = 0; i < Forward::output_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < Forward::output_length) {
            data[i] *= C(symbol[k].x, symbol[k].y);
            if (k == 0 || k == N / 2) data[i].y = 0;
            partial[op * F + k] = make_double2(data[i].x, data[i].y);
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(&g1_arrivals[row], 1u) == 1u;
        if (last) g1_arrivals[row] = g1_ready[row] = 0;
    }
    __syncthreads();
    if (!last) return;
    const double2* other = partial + (op ^ 1) * F;
    for (int i = 0; i < Forward::output_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < Forward::output_length) {
            double2 term = __ldcg(other + k);
            out[row * F + k] = make_double2(-__dadd_rn(data[i].x, term.x), -__dadd_rn(data[i].y, term.y));
        }
    }
}

// Hero M1 analytic term -sum_j L_j[eta L_j xi] from the FP32 trunk input rows (eta_norm, xi_phys): block
// (row, j) transforms FP64-promoted xi and applies operator j. Reproduces the accepted convert, EPT8 cuFFTDx
// FFT, spectral sandwich and XLA reduce exactly. partial is [rows, 2, F].
template <class Forward, class Inverse>
__global__ void g1_kernel(const float* __restrict__ normalized, const double2* __restrict__ operators,
                          double2* __restrict__ out, double2* __restrict__ partial, int n_depth, float eta_scale) {
    using C = typename Forward::value_type;
    constexpr int N = size_of<Forward>::value;
    static_assert(Forward::output_ept == Inverse::input_ept && Forward::stride == Inverse::stride, "shared register layout");
    extern __shared__ __align__(16) unsigned char scratch[];
    const int row = blockIdx.x / 2, op = blockIdx.x % 2;
    const double2* symbol = operators + (row % n_depth * 2 + op) * (N / 2 + 1);
    const float2* eta = reinterpret_cast<const float2*>(normalized) + row * N;
    C data[Forward::storage_size] = {};
    for (int i = 0; i < Forward::input_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < Forward::input_length) {
            float2 xi = eta[N / 2 + k];
            data[i] = C(xi.x, xi.y);
        }
    }
    Forward().execute(data, scratch);
    for (int i = 0; i < Forward::storage_size; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (i < Inverse::input_ept && k < Inverse::input_length) {
            data[i] = data[i] * C(symbol[k].x, symbol[k].y);
            if (k == 0 || k == N / 2) data[i].y = 0;
        } else {
            data[i] = C(0, 0);
        }
    }
    __syncthreads();
    Inverse().execute(data, scratch);
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) {
            float2 surface = eta[k];
            C weight(__fmul_rn(surface.x, eta_scale), __fmul_rn(surface.y, eta_scale));
            data[i] *= double(1.0 / N);
            data[i].x *= weight.x;
            data[i].y *= weight.y;
        }
    }
    g1_finish<Forward>(data, scratch, symbol, partial + row * 2 * (N / 2 + 1), out, row, op);
}

// The same term from the FP64 state spectra, eta = eta_factor * c2r(eta_hat) and xi_factor * xi_hat, with two
// FFTs on the critical path: blocks [0, rows) inverse transform eta and publish it in scratch slot 2, while
// block rows + 2 * row + j inverse transforms L_j xi, then waits for its row's eta. scratch is [rows, 3, F].
template <class Forward, class Inverse>
__global__ void exact_g1_kernel(const double2* __restrict__ eta_spectra, const double2* __restrict__ xi_spectra,
                                const double2* __restrict__ operators, double2* __restrict__ out,
                                double2* __restrict__ scratch_out, int rows, int n_depth, double eta_factor,
                                double xi_factor) {
    using C = typename Forward::value_type;
    constexpr int N = size_of<Forward>::value, F = N / 2 + 1;
    static_assert(Forward::output_ept == Inverse::input_ept && Forward::stride == Inverse::stride, "shared register layout");
    extern __shared__ __align__(16) unsigned char scratch[];
    C data[Forward::storage_size] = {};
    if (blockIdx.x < rows) {
        const double2* eta_hat = eta_spectra + blockIdx.x * F;
        for (int i = 0; i < Inverse::input_ept; ++i) {
            int k = threadIdx.x + i * Inverse::stride;
            if (k < Inverse::input_length) data[i] = C(eta_hat[k].x, k == 0 || k == N / 2 ? 0.0 : eta_hat[k].y);
        }
        Inverse().execute(data, scratch);
        double2* eta = scratch_out + (blockIdx.x * 3 + 2) * F;
        for (int i = 0; i < Inverse::output_ept; ++i) {
            int k = threadIdx.x + i * Inverse::stride;
            if (k < Inverse::output_length) eta[k] = make_double2(data[i].x * eta_factor, data[i].y * eta_factor);
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicExch(&g1_ready[blockIdx.x], 1u);
        return;
    }
    const int row = (blockIdx.x - rows) / 2, op = (blockIdx.x - rows) % 2;
    const double2* symbol = operators + (row % n_depth * 2 + op) * F;
    for (int i = 0; i < Inverse::input_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::input_length) {
            double2 xi = xi_spectra[row * F + k];
            data[i] = C(xi.x * xi_factor, xi.y * xi_factor) * C(symbol[k].x, symbol[k].y);
            if (k == 0 || k == N / 2) data[i].y = 0;
        }
    }
    Inverse().execute(data, scratch);
    if (threadIdx.x == 0)
        while (*reinterpret_cast<volatile unsigned*>(&g1_ready[row]) == 0) {}
    __threadfence();
    __syncthreads();
    const double2* eta = scratch_out + (row * 3 + 2) * F;
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) {
            double2 weight = __ldcg(eta + k);
            data[i] *= double(1.0 / N);
            data[i].x *= weight.x;
            data[i].y *= weight.y;
        }
    }
    g1_finish<Forward>(data, scratch, symbol, scratch_out + row * 3 * F, out, row, op);
}

using G1Forward = decltype(Base64<1024>() + Type<fft_type::r2c>() + Folded());
using G1Inverse = decltype(Base64<1024>() + Type<fft_type::c2r>() + Folded());

ffi::Error g1(cudaStream_t stream, ffi::AnyBuffer normalized, ffi::AnyBuffer operators,
              ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> partial, float eta_scale) {
    const auto dims = normalized.dimensions();
    const int64_t rows = normalized.element_count() / (2 * 1024);
    const int n_depth = static_cast<int>(operators.element_count() / (2 * 513));
    if (dims.back() != 1024 || dims[dims.size() - 2] != 2 || n_depth < 1 || rows % n_depth != 0 || rows > kMaxG1Rows)
        return ffi::Error::InvalidArgument("G1 supports [..., 2, 1024] inputs and [D, 2, 513] operators.");
    g1_kernel<G1Forward, G1Inverse><<<rows * 2, G1Forward::block_dim, g1_shared<G1Forward, G1Inverse>(), stream>>>(
        static_cast<const float*>(normalized.untyped_data()), static_cast<const double2*>(operators.untyped_data()),
        static_cast<double2*>(out->untyped_data()), static_cast<double2*>(partial->untyped_data()), n_depth, eta_scale);
    auto error = cudaGetLastError();
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

ffi::Error exact_g1(cudaStream_t stream, ffi::AnyBuffer eta_hat, ffi::AnyBuffer xi_hat, ffi::AnyBuffer operators,
                    ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> scratch_out, double eta_factor,
                    double xi_factor) {
    const int rows = static_cast<int>(eta_hat.element_count() / 513);
    const int n_depth = static_cast<int>(operators.element_count() / (2 * 513));
    if (eta_hat.dimensions().back() != 513 || xi_hat.element_count() != eta_hat.element_count() || n_depth < 1
            || rows % n_depth != 0 || rows > kMaxG1Rows)
        return ffi::Error::InvalidArgument("Exact G1 supports matching [..., 513] spectra and [D, 2, 513] operators.");
    exact_g1_kernel<G1Forward, G1Inverse><<<rows * 3, G1Forward::block_dim, g1_shared<G1Forward, G1Inverse>(), stream>>>(
        static_cast<const double2*>(eta_hat.untyped_data()), static_cast<const double2*>(xi_hat.untyped_data()),
        static_cast<const double2*>(operators.untyped_data()), static_cast<double2*>(out->untyped_data()),
        static_cast<double2*>(scratch_out->untyped_data()), rows, n_depth, eta_factor, xi_factor);
    auto error = cudaGetLastError();
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

// Hero trunk inputs from the FP64 state spectra: block (row, channel) rounds the spectrum to FP32, inverse
// transforms it, scales by multiplier (1/N, feature scale and xi_scale) into the normalized row, and forward
// transforms that row. Replaces the FP64 inverse FFT, its FP32 normalization and the cuFFT r2c.
template <class Forward, class Inverse>
__global__ void trunk_input_kernel(const double2* __restrict__ eta_hat, const double2* __restrict__ xi_hat,
                                   float* __restrict__ normalized, float2* __restrict__ normalized_hat,
                                   float eta_multiplier, float xi_multiplier) {
    using C = typename Forward::value_type;
    constexpr int N = size_of<Forward>::value, F = N / 2 + 1;
    static_assert(Inverse::output_ept == Forward::input_ept && Forward::stride == Inverse::stride, "shared register layout");
    extern __shared__ __align__(16) unsigned char scratch[];
    const int row = blockIdx.x / 2, channel = blockIdx.x % 2;
    const double2* spectrum = (channel ? xi_hat : eta_hat) + row * F;
    const float multiplier = channel ? xi_multiplier : eta_multiplier;
    C data[Inverse::storage_size] = {};
    for (int i = 0; i < Inverse::input_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::input_length) {
            double2 value = spectrum[k];
            data[i] = C(float(value.x), k == 0 || k == N / 2 ? 0.0f : float(value.y));
        }
    }
    Inverse().execute(data, scratch);
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) {
            data[i] = C(__fmul_rn(data[i].x, multiplier), __fmul_rn(data[i].y, multiplier));
            reinterpret_cast<float2*>(normalized)[blockIdx.x * N / 2 + k] = make_float2(data[i].x, data[i].y);
        }
    }
    __syncthreads();
    Forward().execute(data, scratch);
    for (int i = 0; i < Forward::output_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < Forward::output_length)
            normalized_hat[blockIdx.x * F + k] = make_float2(data[i].x, k == 0 || k == N / 2 ? 0.0f : data[i].y);
    }
}

ffi::Error trunk_input(cudaStream_t stream, ffi::AnyBuffer eta_hat, ffi::AnyBuffer xi_hat,
                       ffi::Result<ffi::AnyBuffer> normalized, ffi::Result<ffi::AnyBuffer> normalized_hat,
                       float eta_multiplier, float xi_multiplier) {
    using Forward = decltype(Base<1024, 8>() + Type<fft_type::r2c>() + Folded());
    using Inverse = decltype(Base<1024, 8>() + Type<fft_type::c2r>() + Folded());
    constexpr int shared = Forward::shared_memory_size > Inverse::shared_memory_size
                           ? Forward::shared_memory_size : Inverse::shared_memory_size;
    if (eta_hat.dimensions().back() != 513 || xi_hat.element_count() != eta_hat.element_count())
        return ffi::Error::InvalidArgument("Trunk inputs support matching [..., 513] state spectra.");
    trunk_input_kernel<Forward, Inverse><<<eta_hat.element_count() / 513 * 2, Forward::block_dim, shared, stream>>>(
        static_cast<const double2*>(eta_hat.untyped_data()), static_cast<const double2*>(xi_hat.untyped_data()),
        static_cast<float*>(normalized->untyped_data()), static_cast<float2*>(normalized_hat->untyped_data()),
        eta_multiplier, xi_multiplier);
    auto error = cudaGetLastError();
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

// Trunk inputs, surface features and branch fields straight from the FP64 state spectra, one FP32 inverse FFT
// per block. Per row: blocks 0-1 write the normalized eta/xi rows (multiplier = 1/N over the feature scale, xi
// also times xi_scale; eta also writes eta, eta^2, eta^3 as features 0-2), blocks 2..2+S-1 the surface-operator
// features of eta (features 3..), and the rest the filtered branch fields of xi. The operators [S, F] and the
// per-trajectory filters [D, R, F] are indexed in place. Replaces trunk_input_kernel and the field inverse.
template <class Inverse>
__global__ void trunk_fields_kernel(const double2* __restrict__ eta_hat, const double2* __restrict__ xi_hat,
                                    const float2* __restrict__ surface_ops, const float* __restrict__ filters,
                                    float* __restrict__ normalized, float* __restrict__ features,
                                    float* __restrict__ fields, int surfaces, int branches, int n_depth,
                                    float eta_multiplier, float xi_multiplier) {
    using C = typename Inverse::value_type;
    constexpr int N = size_of<Inverse>::value, F = N / 2 + 1;
    extern __shared__ __align__(16) unsigned char scratch[];
    const int per_row = 2 + surfaces + branches, row = blockIdx.x / per_row, field = blockIdx.x % per_row;
    const bool from_xi = field == 1 || field >= 2 + surfaces;
    const double2* spectrum = (from_xi ? xi_hat : eta_hat) + row * F;
    C data[Inverse::storage_size] = {};
    for (int i = 0; i < Inverse::input_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::input_length) {
            double2 value = spectrum[k];
            data[i] = C(float(value.x), float(value.y));
            if (field >= 2 + surfaces) {
                float filter = filters[((row % n_depth) * branches + field - 2 - surfaces) * F + k];
                data[i] = C(__fmul_rn(data[i].x, filter), __fmul_rn(data[i].y, filter));
            } else if (field >= 2) {
                float2 op = surface_ops[(field - 2) * F + k];
                data[i] = data[i] * C(op.x, op.y);
            }
            if (k == 0 || k == N / 2) data[i].y = 0;
        }
    }
    Inverse().execute(data, scratch);
    const float multiplier = from_xi ? xi_multiplier : eta_multiplier;
    float2* out = reinterpret_cast<float2*>(
        field < 2 ? normalized + (row * 2 + field) * N
        : field < 2 + surfaces ? features + (row * (surfaces + 3) + field + 1) * N
        : fields + (row * branches + field - 2 - surfaces) * N);
    float2* powers = reinterpret_cast<float2*>(features + row * (surfaces + 3) * N);
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) {
            float2 value = make_float2(__fmul_rn(data[i].x, multiplier), __fmul_rn(data[i].y, multiplier));
            out[k] = value;
            if (field == 0) {
                float2 square = make_float2(__fmul_rn(value.x, value.x), __fmul_rn(value.y, value.y));
                powers[k] = value;
                powers[N / 2 + k] = square;
                powers[N + k] = make_float2(__fmul_rn(square.x, value.x), __fmul_rn(square.y, value.y));
            }
        }
    }
}

ffi::Error trunk_fields(cudaStream_t stream, ffi::AnyBuffer eta_hat, ffi::AnyBuffer xi_hat, ffi::AnyBuffer surface_ops,
                        ffi::AnyBuffer filters, ffi::Result<ffi::AnyBuffer> normalized,
                        ffi::Result<ffi::AnyBuffer> features, ffi::Result<ffi::AnyBuffer> fields,
                        float eta_multiplier, float xi_multiplier) {
    using Inverse = decltype(Base<1024, 8>() + Type<fft_type::c2r>() + Folded());
    const int64_t rows = eta_hat.element_count() / 513;
    const int surfaces = static_cast<int>(surface_ops.element_count() / 513);
    const auto filter_dims = filters.dimensions();
    const int branches = static_cast<int>(filter_dims[filter_dims.size() - 2]);
    const int n_depth = static_cast<int>(filters.element_count() / (int64_t(branches) * 513));
    if (eta_hat.dimensions().back() != 513 || xi_hat.element_count() != eta_hat.element_count()
            || filter_dims.back() != 513 || n_depth < 1 || rows % n_depth != 0)
        return ffi::Error::InvalidArgument("Trunk fields support [..., 513] spectra and [D, R, 513] filters.");
    trunk_fields_kernel<Inverse><<<rows * (2 + surfaces + branches), Inverse::block_dim, Inverse::shared_memory_size, stream>>>(
        static_cast<const double2*>(eta_hat.untyped_data()), static_cast<const double2*>(xi_hat.untyped_data()),
        static_cast<const float2*>(surface_ops.untyped_data()), static_cast<const float*>(filters.untyped_data()),
        static_cast<float*>(normalized->untyped_data()), static_cast<float*>(features->untyped_data()),
        static_cast<float*>(fields->untyped_data()), surfaces, branches, n_depth, eta_multiplier, xi_multiplier);
    auto error = cudaGetLastError();
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

template <int EPT>
ffi::Error launch(cudaStream_t stream, const float* spectrum_input, const float* context_symbol,
                  const float* inner_filters, const float* gamma, const float* normalized, float* context,
                  float2* nested, int64_t stages, int channels, int n_nested, int n_depth, float nested_reciprocal) {
    using Forward = decltype(Base<1024, EPT>() + Type<fft_type::r2c>() + Folded());
    using Inverse = decltype(Base<1024, EPT>() + Type<fft_type::c2r>() + Folded());
    constexpr int shared = Forward::shared_memory_size > Inverse::shared_memory_size
                           ? Forward::shared_memory_size : Inverse::shared_memory_size;
    context_filter_kernel<Forward, Inverse><<<stages * (channels + n_nested), Forward::block_dim, shared, stream>>>(
        spectrum_input, context_symbol, inner_filters, gamma, normalized, context, nested, channels, n_nested,
        n_depth, nested_reciprocal);
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
    const int n_depth = static_cast<int>(context_symbol.element_count() / (int64_t(points / 2 + 1) * channels));
    if (points != 1024 || rows != channels + n_nested || n_depth < 1 || stages % n_depth != 0
            || inner_filters.element_count() != int64_t(n_depth) * n_nested * (points / 2 + 1))
        return ffi::Error::InvalidArgument("Context filter supports 1024 points and context+nested rows.");
    auto run = ept == 4 ? launch<4> : ept == 16 ? launch<16> : launch<8>;
    return run(stream, static_cast<const float*>(spectrum_input.untyped_data()),
               static_cast<const float*>(context_symbol.untyped_data()),
               static_cast<const float*>(inner_filters.untyped_data()), static_cast<const float*>(gamma.untyped_data()),
               static_cast<const float*>(normalized.untyped_data()), static_cast<float*>(context->untyped_data()),
               static_cast<float2*>(nested->untyped_data()), stages, channels, n_nested, n_depth, nested_reciprocal);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_context_filter, context_filter,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Attr<float>("nested_reciprocal").Attr<int64_t>("ept"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_g1, g1,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Attr<float>("eta_scale"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_exact_g1, exact_g1,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Attr<double>("eta_factor").Attr<double>("xi_factor"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_trunk_input, trunk_input,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Attr<float>("eta_multiplier").Attr<float>("xi_multiplier"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_trunk_fields, trunk_fields,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Attr<float>("eta_multiplier").Attr<float>("xi_multiplier"),
    {ffi::Traits::kCmdBufferCompatible});
