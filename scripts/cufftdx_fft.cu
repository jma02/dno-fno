// Inference-only cuFFTDx experiment; no changes to the production FFT backend.
#include <cuda_runtime.h>
#include <cufftdx.hpp>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace cufftdx;

#ifndef DNO_SM
#define DNO_SM 890
#endif

template<int N, class T, int EPT>
using FFTBase = decltype(Block() + Size<N>() + Precision<T>() + SM<DNO_SM>() +
                         ElementsPerThread<EPT>() + FFTsPerBlock<1>());

template<class FFT, bool outer = false>
__global__ void transform(const void* a, const void* b, void* out,
                          const void* symbols, int input_length, bool multiply) {
    using C = typename FFT::value_type;
    using I = typename FFT::input_type;
    using O = typename FFT::output_type;
    constexpr bool inverse = type_of<FFT>::value == fft_type::c2r;
    C data[FFT::storage_size] = {};
    extern __shared__ __align__(16) unsigned char scratch[];
    for (int i = 0; i < FFT::input_ept; ++i) {
        int k = threadIdx.x + i * FFT::stride;
        if (k < FFT::input_length && k < input_length) {
            I value = static_cast<const I*>(a)[blockIdx.x * input_length + k];
            if (multiply) {
                I weight = static_cast<const I*>(b)[blockIdx.x * input_length + k];
                value.x *= weight.x;
                value.y *= weight.y;
            }
            if constexpr (inverse) {
                if (k == 0 || k == size_of<FFT>::value / 2) value.y = 0;
            }
            reinterpret_cast<I*>(data)[i] = value;
        }
    }
    FFT().execute(data, scratch);
    for (int i = 0; i < FFT::output_ept; ++i) {
        int k = threadIdx.x + i * FFT::stride;
        if (k < FFT::output_length) {
            O value = reinterpret_cast<O*>(data)[i];
            if constexpr (inverse) value *= precision_of_t<FFT>(1.0 / size_of<FFT>::value);
            if constexpr (outer) {
                value *= precision_of_t<FFT>(1.0 / size_of<FFT>::value);
                value *= static_cast<const C*>(symbols)[blockIdx.x * FFT::output_length + k];
            }
            static_cast<O*>(out)[blockIdx.x * FFT::output_length + k] = value;
        }
    }
}

template<class Forward, class Inverse>
__global__ void sandwich_kernel(const void* eta, const void* xi, const void* symbols,
                                void* out, int branches) {
    using C = typename Forward::value_type;
    constexpr int N = size_of<Forward>::value;
    int batch = blockIdx.x / branches;
    C data[Forward::storage_size] = {};
    extern __shared__ __align__(16) unsigned char scratch[];
    for (int i = 0; i < Forward::input_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < Forward::input_length) data[i] = static_cast<const C*>(xi)[batch * N / 2 + k];
    }
    // L [eta L xi], with normalization fused into the physical-space products.
    for (int stage = 0; stage < 2; ++stage) {
        Forward().execute(data, scratch);
        for (int i = 0; i < Forward::output_ept; ++i) {
            int k = threadIdx.x + i * Forward::stride;
            if (k < Forward::output_length) {
                data[i] *= static_cast<const C*>(symbols)[blockIdx.x * (N / 2 + 1) + k];
                if (k == 0 || k == N / 2) data[i].y = 0;
            }
        }
        __syncthreads();
        Inverse().execute(data, scratch);
        for (int i = 0; i < Inverse::output_ept; ++i) {
            int k = threadIdx.x + i * Inverse::stride;
            if (k < Inverse::output_length) {
                data[i] *= precision_of_t<Forward>(1.0 / N);
                if (stage == 0) {
                    C weight = static_cast<const C*>(eta)[batch * N / 2 + k];
                    data[i].x *= weight.x;
                    data[i].y *= weight.y;
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) static_cast<C*>(out)[blockIdx.x * N / 2 + k] = data[i];
    }
}

template<class Forward, class Inverse>
__global__ void joint_kernel(const void* eta, const void* xi, const void* symbols, void* out) {
    using C = typename Forward::value_type;
    constexpr int N = size_of<Forward>::value;
    C data[Forward::storage_size] = {}, input[Forward::storage_size], sum[Forward::storage_size] = {};
    extern __shared__ __align__(16) unsigned char scratch[];
    for (int i = 0; i < Forward::input_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < Forward::input_length) data[i] = static_cast<const C*>(xi)[blockIdx.x * N / 2 + k];
    }
    Forward().execute(data, scratch);
    for (int i = 0; i < Forward::storage_size; ++i) input[i] = data[i];
    for (int branch = 0; branch < 2; ++branch) {
        for (int i = 0; i < Forward::output_ept; ++i) {
            int k = threadIdx.x + i * Forward::stride;
            if (k < Forward::output_length) {
                data[i] = input[i] * static_cast<const C*>(symbols)[(2 * blockIdx.x + branch) * (N / 2 + 1) + k];
                if (k == 0 || k == N / 2) data[i].y = 0;
            }
        }
        __syncthreads();
        Inverse().execute(data, scratch);
        for (int i = 0; i < Inverse::output_ept; ++i) {
            int k = threadIdx.x + i * Inverse::stride;
            if (k < Inverse::output_length) {
                C weight = static_cast<const C*>(eta)[blockIdx.x * N / 2 + k];
                data[i] *= precision_of_t<Forward>(1.0 / N);
                data[i].x *= weight.x;
                data[i].y *= weight.y;
            }
        }
        __syncthreads();
        Forward().execute(data, scratch);
        for (int i = 0; i < Forward::output_ept; ++i) {
            int k = threadIdx.x + i * Forward::stride;
            if (k < Forward::output_length)
                sum[i] += data[i] * static_cast<const C*>(symbols)[(2 * blockIdx.x + branch) * (N / 2 + 1) + k];
        }
    }
    for (int i = 0; i < Forward::storage_size; ++i) data[i] = sum[i];
    for (int i = 0; i < Forward::output_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k == 0 || k == N / 2) data[i].y = 0;
    }
    __syncthreads();
    Inverse().execute(data, scratch);
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) {
            data[i] *= precision_of_t<Forward>(1.0 / N);
            static_cast<C*>(out)[blockIdx.x * N / 2 + k] = data[i];
        }
    }
}

template<int N, class T, int EPT>
ffi::Error launch(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                  ffi::AnyBuffer symbols, ffi::Result<ffi::AnyBuffer> out,
                  int64_t operation, bool multiply) {
    using Options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using F = decltype(FFTBase<N, T, EPT>() + Type<fft_type::r2c>() + Options());
    using I = decltype(FFTBase<N, T, EPT>() + Type<fft_type::c2r>() + Options());
    constexpr int shared = F::shared_memory_size > I::shared_memory_size ? F::shared_memory_size : I::shared_memory_size;
    if constexpr (shared > 48 * 1024) {
        static auto first = cudaFuncSetAttribute(transform<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
        static auto second = cudaFuncSetAttribute(transform<I>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
        if (first != cudaSuccess || second != cudaSuccess)
            return ffi::Error::Internal("Cannot reserve cuFFTDx shared memory.");
    }
    if (operation == 2 || operation == 3) {
        if constexpr (N == 1024) {
            int blocks = out->element_count() / N;
            if (operation == 3) {
                joint_kernel<F, I><<<blocks, F::block_dim, shared, stream>>>(
                    a.untyped_data(), b.untyped_data(), symbols.untyped_data(), out->untyped_data());
            } else {
                int branches = symbols.dimensions()[symbols.dimensions().size() - 2];
                sandwich_kernel<F, I><<<blocks, F::block_dim, shared, stream>>>(
                    a.untyped_data(), b.untyped_data(), symbols.untyped_data(), out->untyped_data(), branches);
            }
        } else {
            return ffi::Error::InvalidArgument("Fused G1 supports the 1024-point model grid only.");
        }
    } else if (operation == 1) {
        int length = a.dimensions().back();
        transform<I><<<out->element_count() / N, I::block_dim, I::shared_memory_size, stream>>>(
            a.untyped_data(), b.untyped_data(), out->untyped_data(), symbols.untyped_data(), length, false);
    } else if (operation == 4) {
        transform<F, true><<<out->element_count() / F::output_length, F::block_dim, F::shared_memory_size, stream>>>(
            a.untyped_data(), b.untyped_data(), out->untyped_data(), symbols.untyped_data(), N / 2, true);
    } else {
        int length = a.dimensions().back() / 2;
        transform<F><<<a.element_count() / (2 * length), F::block_dim, F::shared_memory_size, stream>>>(
            a.untyped_data(), b.untyped_data(), out->untyped_data(), symbols.untyped_data(), length, multiply);
    }
    auto error = cudaGetLastError();
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

ffi::Error execute(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                    ffi::AnyBuffer symbols, ffi::Result<ffi::AnyBuffer> out,
                    int64_t size, int64_t operation, bool multiply, int64_t ept) {
    bool fp64 = a.element_type() == ffi::F64 || a.element_type() == ffi::C128;
    if (size == 1024) {
        if (ept == 4)
            return fp64 ? launch<1024, double, 4>(stream, a, b, symbols, out, operation, multiply)
                        : launch<1024, float, 4>(stream, a, b, symbols, out, operation, multiply);
        if (ept == 16)
            return fp64 ? launch<1024, double, 16>(stream, a, b, symbols, out, operation, multiply)
                        : launch<1024, float, 16>(stream, a, b, symbols, out, operation, multiply);
        return fp64 ? launch<1024, double, 8>(stream, a, b, symbols, out, operation, multiply)
                    : launch<1024, float, 8>(stream, a, b, symbols, out, operation, multiply);
    }
    if (size == 8192) {
        return fp64 ? launch<8192, double, 8>(stream, a, b, symbols, out, operation, multiply)
                    : launch<8192, float, 8>(stream, a, b, symbols, out, operation, multiply);
    }
    return ffi::Error::InvalidArgument("Benchmark supports FFT lengths 1024 and 8192.");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_cufftdx, execute,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Attr<int64_t>("size").Attr<int64_t>("operation").Attr<bool>("multiply").Attr<int64_t>("ept"),
    {ffi::Traits::kCmdBufferCompatible});
