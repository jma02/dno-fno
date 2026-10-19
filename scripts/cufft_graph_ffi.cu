// Opt-in cuFFT execution through capture-compatible typed JAX FFI.
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory>
#include <mutex>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

struct Plan {
    static ffi::TypeId id;
    cufftHandle handle = 0;
    cufftType kind;
    int device = 0;
    size_t copy_bytes = 0;
    size_t workspace_offset = 0;
    std::mutex mutex;

    ~Plan() {
        if (!handle) return;
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        cufftDestroy(handle);
        if (previous != device) cudaSetDevice(previous);
    }
};

ffi::TypeId Plan::id = {};
extern "C" void* dno_cufft_type_id() { return &Plan::id; }
extern "C" void* dno_cufft_type_info() {
    static auto info = ffi::MakeTypeInfo<Plan>();
    return &info;
}

ffi::Error cufft_error(const char* operation, cufftResult status) {
    return ffi::Error::Internal(std::string(operation) + ": cuFFT status " + std::to_string(status));
}

ffi::ErrorOr<std::unique_ptr<Plan>> initialize(
    ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::Buffer<ffi::U8>> scratch, int64_t n, bool inverse) {
    auto plan = std::make_unique<Plan>();
    auto cuda_status = cudaGetDevice(&plan->device);
    if (cuda_status != cudaSuccess)
        return ffi::Unexpected(ffi::Error::Internal(cudaGetErrorString(cuda_status)));
    const bool fp64 = input.element_type() == ffi::F64 || input.element_type() == ffi::C128;
    plan->kind = inverse ? (fp64 ? CUFFT_Z2D : CUFFT_C2R) : (fp64 ? CUFFT_D2Z : CUFFT_R2C);
    long long length = n, input_length = inverse ? n / 2 + 1 : n;
    long long output_length = inverse ? n : n / 2 + 1;
    long long batch = input.element_count() / input_length;
    plan->copy_bytes = inverse ? input.size_bytes() : 0;
    plan->workspace_offset = (plan->copy_bytes + 255) / 256 * 256;
    auto status = cufftCreate(&plan->handle);
    if (status != CUFFT_SUCCESS) return ffi::Unexpected(cufft_error("cufftCreate", status));
    status = cufftSetAutoAllocation(plan->handle, 0);
    if (status != CUFFT_SUCCESS) return ffi::Unexpected(cufft_error("cufftSetAutoAllocation", status));
    size_t workspace_bytes = 0;
    status = cufftMakePlanMany64(plan->handle, 1, &length, &input_length, 1, input_length,
                               &output_length, 1, output_length, plan->kind, batch, &workspace_bytes);
    if (status != CUFFT_SUCCESS) return ffi::Unexpected(cufft_error("cufftMakePlanMany64", status));
    if (plan->workspace_offset + workspace_bytes > scratch->size_bytes())
        return ffi::Unexpected(ffi::Error::Internal("cuFFT workspace exceeds the supplied scratch buffer"));
    return std::move(plan);
}

ffi::Error execute(cudaStream_t stream, Plan* plan, ffi::AnyBuffer input,
                   ffi::Result<ffi::AnyBuffer> output,
                   ffi::Result<ffi::Buffer<ffi::U8>> scratch,
                   int64_t /*n*/, bool /*inverse*/) {
    // The mutex protects host-side plan mutation; each execution's workspace is
    // an XLA-owned result buffer, so concurrent streams/graph replays do not race.
    std::lock_guard<std::mutex> lock(plan->mutex);
    auto status = cufftSetStream(plan->handle, stream);
    if (status != CUFFT_SUCCESS) return cufft_error("cufftSetStream", status);
    auto* bytes = scratch->typed_data();
    status = cufftSetWorkArea(plan->handle, bytes + plan->workspace_offset);
    if (status != CUFFT_SUCCESS) return cufft_error("cufftSetWorkArea", status);
    void* source = input.untyped_data();
    if (plan->copy_bytes) {
        // Out-of-place C2R is allowed to overwrite its input. Never give cuFFT
        // an immutable JAX operand; the copy is itself graph-capturable.
        auto error = cudaMemcpyAsync(bytes, source, plan->copy_bytes, cudaMemcpyDeviceToDevice, stream);
        if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
        source = bytes;
    }
    switch (plan->kind) {
        case CUFFT_R2C:
            status = cufftExecR2C(plan->handle, static_cast<cufftReal*>(source),
                                 static_cast<cufftComplex*>(output->untyped_data()));
            break;
        case CUFFT_D2Z:
            status = cufftExecD2Z(plan->handle, static_cast<cufftDoubleReal*>(source),
                                 static_cast<cufftDoubleComplex*>(output->untyped_data()));
            break;
        case CUFFT_C2R:
            status = cufftExecC2R(plan->handle, static_cast<cufftComplex*>(source),
                                 static_cast<cufftReal*>(output->untyped_data()));
            break;
        case CUFFT_Z2D:
            status = cufftExecZ2D(plan->handle, static_cast<cufftDoubleComplex*>(source),
                                 static_cast<cufftDoubleReal*>(output->untyped_data()));
            break;
        default:
            return ffi::Error::Internal("Unsupported cuFFT plan type");
    }
    return status == CUFFT_SUCCESS ? ffi::Error::Success() : cufft_error("cufftExec", status);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_cufft_initialize, initialize,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::Buffer<ffi::U8>>()
        .Attr<int64_t>("n").Attr<bool>("inverse"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_cufft_execute, execute,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::Initialized<Plan>>()
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::Buffer<ffi::U8>>()
        .Attr<int64_t>("n").Attr<bool>("inverse"),
    {ffi::Traits::kCmdBufferCompatible});
