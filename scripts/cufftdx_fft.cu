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

template<class FFT, bool outer = false, bool normalize = false, bool promote = false, bool real_outer = false>
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
            I value;
            if constexpr (promote) {
                float2 input = static_cast<const float2*>(a)[blockIdx.x * input_length + k];
                value = I(input.x, input.y);
            } else {
                value = static_cast<const I*>(a)[blockIdx.x * input_length + k];
            }
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
            if constexpr (real_outer)
                value *= static_cast<const precision_of_t<FFT>*>(symbols)[blockIdx.x * FFT::output_length + k];
            if constexpr (normalize) {
                const float* scales = static_cast<const float*>(b);
                int channel = blockIdx.x % 2;
                float2 normalized = {__fmul_rn(float(value.x), scales[channel]),
                                     __fmul_rn(float(value.y), scales[channel])};
                if (channel == 1) {
                    normalized.x = __fmul_rn(normalized.x, scales[2]);
                    normalized.y = __fmul_rn(normalized.y, scales[2]);
                }
                static_cast<float2*>(out)[blockIdx.x * FFT::output_length + k] = normalized;
            } else {
                static_cast<O*>(out)[blockIdx.x * FFT::output_length + k] = value;
            }
        }
    }
}

template<class Inverse, bool features = false, bool balanced = false>
__global__ void field_inverse_kernel(const void* inputs, const void* surface_ops,
                                     const void* filters, void* out, int surfaces, int branches,
                                     void* feature_out = nullptr, const void* normalized_inputs = nullptr) {
    using C = typename Inverse::value_type;
    using T = precision_of_t<Inverse>;
    constexpr int N = size_of<Inverse>::value;
    constexpr int F = N / 2 + 1;
    int fields = surfaces + branches;
    int batch = blockIdx.x / fields, field = blockIdx.x % fields;
    C data[Inverse::storage_size] = {};
    extern __shared__ __align__(16) unsigned char scratch[];
    for (int i = 0; i < Inverse::input_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::input_length) {
            data[i] = static_cast<const C*>(inputs)[(2 * batch + (field >= surfaces)) * F + k];
            if (field < surfaces) {
                data[i] *= static_cast<const C*>(surface_ops)[field * F + k];
            } else {
                data[i] *= static_cast<const T*>(filters)[(batch * branches + field - surfaces) * F + k];
            }
            if (k == 0 || k == N / 2) data[i].y = 0;
        }
    }
    Inverse().execute(data, scratch);
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) {
            data[i] *= T(1.0 / N);
            if constexpr (features) {
                if (field < surfaces) {
                    C* output = static_cast<C*>(feature_out);
                    int index = batch * (surfaces + 3) * N / 2 + k;
                    output[index + (3 + field) * N / 2] = data[i];
                    if constexpr (balanced) {
                        if (field < 3) {
                            C value = static_cast<const C*>(normalized_inputs)[batch * N + k];
                            C power = value;
                            if (field > 0) power = C(__fmul_rn(value.x, value.x), __fmul_rn(value.y, value.y));
                            if (field == 2) power = C(__fmul_rn(power.x, value.x), __fmul_rn(power.y, value.y));
                            output[index + field * N / 2] = power;
                        }
                    } else if (field == 0) {
                        C value = static_cast<const C*>(normalized_inputs)[batch * N + k];
                        T square_x = __fmul_rn(value.x, value.x), square_y = __fmul_rn(value.y, value.y);
                        output[index] = value;
                        output[index + N / 2] = C(square_x, square_y);
                        output[index + N] = C(__fmul_rn(square_x, value.x), __fmul_rn(square_y, value.y));
                    }
                } else {
                    static_cast<C*>(out)[(batch * branches + field - surfaces) * N / 2 + k] = data[i];
                }
            } else {
                static_cast<C*>(out)[blockIdx.x * Inverse::output_length + k] = data[i];
            }
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

template<class Forward, class Inverse, bool prepare_eta = false>
__global__ void spectral_sandwich_kernel(const void* eta, const void* xi_hat,
                                         const void* symbols, void* out, int branches) {
    using C = typename Forward::value_type;
    using T = precision_of_t<Forward>;
    constexpr int N = size_of<Forward>::value;
    constexpr int F = N / 2 + 1;
    constexpr int symbol_length = F + prepare_eta;
    int batch = blockIdx.x / branches;
    C data[Forward::storage_size] = {};
    extern __shared__ __align__(16) unsigned char scratch[];
    for (int i = 0; i < Inverse::input_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::input_length) {
            data[i] = static_cast<const C*>(xi_hat)[batch * F + k]
                    * static_cast<const C*>(symbols)[blockIdx.x * symbol_length + k];
            if (k == 0 || k == N / 2) data[i].y = 0;
        }
    }
    Inverse().execute(data, scratch);
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) {
            C weight;
            if constexpr (prepare_eta) {
                float2 input = static_cast<const float2*>(eta)[batch * N / 2 + k];
                float scale = float(static_cast<const C*>(symbols)[blockIdx.x * symbol_length + F].x);
                weight = C(__fmul_rn(input.x, scale), __fmul_rn(input.y, scale));
            } else {
                weight = static_cast<const C*>(eta)[batch * N / 2 + k];
            }
            data[i] *= T(1.0 / N);
            data[i].x *= weight.x;
            data[i].y *= weight.y;
        }
    }
    __syncthreads();
    Forward().execute(data, scratch);
    for (int i = 0; i < Forward::output_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < Forward::output_length) {
            data[i] *= static_cast<const C*>(symbols)[blockIdx.x * symbol_length + k];
            if (k == 0 || k == N / 2) data[i].y = 0;
            static_cast<C*>(out)[blockIdx.x * F + k] = data[i];
        }
    }
}

template<class Forward, class Inverse>
__global__ void branch_kernel(const void* xi_hat, const void* weights, const void* filters,
                             void* out, int branches) {
    using C = typename Forward::value_type;
    using T = precision_of_t<Forward>;
    constexpr int N = size_of<Forward>::value;
    constexpr int F = N / 2 + 1;
    int batch = blockIdx.x / branches, branch = blockIdx.x % branches;
    C data[Forward::storage_size] = {};
    extern __shared__ __align__(16) unsigned char scratch[];
    for (int i = 0; i < Inverse::input_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::input_length) {
            data[i] = static_cast<const C*>(xi_hat)[batch * F + k];
            data[i] *= static_cast<const T*>(filters)[(batch * F + k) * branches + branch];
            if (k == 0 || k == N / 2) data[i].y = 0;
        }
    }
    Inverse().execute(data, scratch);
    for (int i = 0; i < Inverse::output_ept; ++i) {
        int k = threadIdx.x + i * Inverse::stride;
        if (k < Inverse::output_length) {
            int offset = (batch * N + 2 * k) * branches + branch;
            data[i] *= T(1.0 / N);
            data[i].x *= static_cast<const T*>(weights)[offset];
            data[i].y *= static_cast<const T*>(weights)[offset + branches];
        }
    }
    __syncthreads();
    Forward().execute(data, scratch);
    for (int i = 0; i < Forward::output_ept; ++i) {
        int k = threadIdx.x + i * Forward::stride;
        if (k < Forward::output_length) {
            data[i] *= T(1.0 / N);
            data[i] *= static_cast<const T*>(filters)[(batch * F + k) * branches + branch];
            static_cast<C*>(out)[blockIdx.x * F + k] = data[i];
        }
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
    if (operation == 11) {
        // The logical [B,N,R] head is supplied column-major, already contiguous along N.
        transform<F, false, false, false, true><<<out->element_count() / F::output_length, F::block_dim, F::shared_memory_size, stream>>>(
            a.untyped_data(), b.untyped_data(), out->untyped_data(), symbols.untyped_data(), N / 2, true);
    } else if (operation == 9) {
        if constexpr (N == 1024) {
            transform<F, false, false, true><<<out->element_count() / F::output_length, F::block_dim, F::shared_memory_size, stream>>>(
                a.untyped_data(), b.untyped_data(), out->untyped_data(), symbols.untyped_data(), N / 2, false);
        } else {
            return ffi::Error::InvalidArgument("Promoted input FFT supports the 1024-point model grid only.");
        }
    } else if (operation == 8) {
        if constexpr (N == 1024) {
            transform<I, false, true><<<out->element_count() / N, I::block_dim, I::shared_memory_size, stream>>>(
                a.untyped_data(), b.untyped_data(), out->untyped_data(), symbols.untyped_data(), N / 2 + 1, false);
        } else {
            return ffi::Error::InvalidArgument("Input normalization supports the 1024-point model grid only.");
        }
    } else if (operation == 7) {
        if constexpr (N == 1024) {
            int surfaces = b.dimensions()[b.dimensions().size() - 2];
            int branches = symbols.dimensions()[symbols.dimensions().size() - 2];
            field_inverse_kernel<I><<<out->element_count() / N, I::block_dim, I::shared_memory_size, stream>>>(
                a.untyped_data(), b.untyped_data(), symbols.untyped_data(), out->untyped_data(), surfaces, branches);
        } else {
            return ffi::Error::InvalidArgument("Field inverse FFT supports the 1024-point model grid only.");
        }
    } else if (operation == 6 || operation == 10) {
        if constexpr (N == 1024) {
            int branches = out->dimensions()[out->dimensions().size() - 2];
            if (operation == 10) {
                spectral_sandwich_kernel<F, I, true><<<out->element_count() / F::output_length, F::block_dim, shared, stream>>>(
                    a.untyped_data(), b.untyped_data(), symbols.untyped_data(), out->untyped_data(), branches);
            } else {
                spectral_sandwich_kernel<F, I><<<out->element_count() / F::output_length, F::block_dim, shared, stream>>>(
                    a.untyped_data(), b.untyped_data(), symbols.untyped_data(), out->untyped_data(), branches);
            }
        } else {
            return ffi::Error::InvalidArgument("Spectral G1 supports the 1024-point model grid only.");
        }
    } else if (operation == 5) {
        if constexpr (N == 1024) {
            int branches = b.dimensions().back();
            branch_kernel<F, I><<<out->element_count() / F::output_length, F::block_dim, shared, stream>>>(
                a.untyped_data(), b.untyped_data(), symbols.untyped_data(), out->untyped_data(), branches);
        } else {
            return ffi::Error::InvalidArgument("Fused learned branches support the 1024-point model grid only.");
        }
    } else if (operation == 2 || operation == 3) {
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
    bool fp64 = a.element_type() == ffi::F64 || a.element_type() == ffi::C128 || operation == 9 || operation == 10;
    if (size == 512) {
        if (ept == 4)
            return fp64 ? launch<512, double, 4>(stream, a, b, symbols, out, operation, multiply)
                        : launch<512, float, 4>(stream, a, b, symbols, out, operation, multiply);
        if (ept == 16)
            return fp64 ? launch<512, double, 16>(stream, a, b, symbols, out, operation, multiply)
                        : launch<512, float, 16>(stream, a, b, symbols, out, operation, multiply);
        return fp64 ? launch<512, double, 8>(stream, a, b, symbols, out, operation, multiply)
                    : launch<512, float, 8>(stream, a, b, symbols, out, operation, multiply);
    }
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
    if (size == 2048) {
        if (ept == 4)
            return fp64 ? launch<2048, double, 4>(stream, a, b, symbols, out, operation, multiply)
                        : launch<2048, float, 4>(stream, a, b, symbols, out, operation, multiply);
        if (ept == 16)
            return fp64 ? launch<2048, double, 16>(stream, a, b, symbols, out, operation, multiply)
                        : launch<2048, float, 16>(stream, a, b, symbols, out, operation, multiply);
        return fp64 ? launch<2048, double, 8>(stream, a, b, symbols, out, operation, multiply)
                    : launch<2048, float, 8>(stream, a, b, symbols, out, operation, multiply);
    }
    if (size == 8192) {
        return fp64 ? launch<8192, double, 8>(stream, a, b, symbols, out, operation, multiply)
                    : launch<8192, float, 8>(stream, a, b, symbols, out, operation, multiply);
    }
    return ffi::Error::InvalidArgument("Benchmark supports FFT lengths 512, 1024, 2048 and 8192.");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_cufftdx, execute,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Attr<int64_t>("size").Attr<int64_t>("operation").Attr<bool>("multiply").Attr<int64_t>("ept"),
    {ffi::Traits::kCmdBufferCompatible});

template<int N, int EPT, bool balanced = false>
ffi::Error launch_features(cudaStream_t stream, ffi::AnyBuffer inputs, ffi::AnyBuffer surface_ops,
                           ffi::AnyBuffer filters, ffi::AnyBuffer normalized, ffi::Result<ffi::AnyBuffer> features,
                           ffi::Result<ffi::AnyBuffer> branches) {
    using I = decltype(FFTBase<N, float, EPT>() + Type<fft_type::c2r>() +
                       RealFFTOptions<complex_layout::natural, real_mode::folded>());
    int surfaces = surface_ops.dimensions()[surface_ops.dimensions().size() - 2];
    int count = filters.dimensions()[filters.dimensions().size() - 2];
    field_inverse_kernel<I, true, balanced><<<normalized.element_count() / (2 * N) * (surfaces + count),
                                    I::block_dim, I::shared_memory_size, stream>>>(
        inputs.untyped_data(), surface_ops.untyped_data(), filters.untyped_data(),
        branches->untyped_data(), surfaces, count, features->untyped_data(), normalized.untyped_data());
    auto error = cudaGetLastError();
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

template<int N>
ffi::Error dispatch_features(cudaStream_t stream, ffi::AnyBuffer inputs, ffi::AnyBuffer surface_ops,
                             ffi::AnyBuffer filters, ffi::AnyBuffer normalized, ffi::Result<ffi::AnyBuffer> features,
                             ffi::Result<ffi::AnyBuffer> branches, int64_t ept, bool balanced) {
    if (balanced) {
        if (ept == 4) return launch_features<N, 4, true>(stream, inputs, surface_ops, filters, normalized, features, branches);
        if (ept == 16) return launch_features<N, 16, true>(stream, inputs, surface_ops, filters, normalized, features, branches);
        return launch_features<N, 8, true>(stream, inputs, surface_ops, filters, normalized, features, branches);
    }
    if (ept == 4) return launch_features<N, 4>(stream, inputs, surface_ops, filters, normalized, features, branches);
    if (ept == 16) return launch_features<N, 16>(stream, inputs, surface_ops, filters, normalized, features, branches);
    return launch_features<N, 8>(stream, inputs, surface_ops, filters, normalized, features, branches);
}

ffi::Error execute_features(cudaStream_t stream, ffi::AnyBuffer inputs, ffi::AnyBuffer surface_ops,
                            ffi::AnyBuffer filters, ffi::AnyBuffer normalized, ffi::Result<ffi::AnyBuffer> features,
                            ffi::Result<ffi::AnyBuffer> branches, int64_t ept, bool balanced) {
    const auto size = normalized.dimensions().back();
    if (size == 512)
        return dispatch_features<512>(stream, inputs, surface_ops, filters, normalized, features, branches, ept, balanced);
    if (size == 1024)
        return dispatch_features<1024>(stream, inputs, surface_ops, filters, normalized, features, branches, ept, balanced);
    return ffi::Error::InvalidArgument("Fused features support FFT lengths 512 and 1024.");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_cufftdx_features, execute_features,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Attr<int64_t>("ept").Attr<bool>("balanced"),
    {ffi::Traits::kCmdBufferCompatible});
