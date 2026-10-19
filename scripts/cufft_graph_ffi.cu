// Opt-in cuFFT execution through capture-compatible typed JAX FFI.
#include <cuda_runtime.h>
#include <cufft.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

struct Plan {
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

struct PlanCache {
    static ffi::TypeId id;
    std::mutex mutex;
    std::map<int32_t, std::unique_ptr<Plan>> devices;
};

ffi::TypeId PlanCache::id = {};
extern "C" void* dno_cufft_type_id() { return &PlanCache::id; }
extern "C" void* dno_cufft_type_info() {
    static auto info = ffi::MakeTypeInfo<PlanCache>();
    return &info;
}

ffi::Error cufft_error(const char* operation, cufftResult status) {
    return ffi::Error::Internal(std::string(operation) + ": cuFFT status " + std::to_string(status));
}

ffi::ErrorOr<std::unique_ptr<PlanCache>> instantiate(int64_t /*n*/, bool /*inverse*/) {
    // No CUDA calls here: instantiation is device-independent and executable-lived.
    return std::make_unique<PlanCache>();
}

ffi::Error initialize(
    PlanCache* cache, int32_t device, ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::Buffer<ffi::U8>> scratch, int64_t n, bool inverse) {
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (cache->devices.count(device)) return ffi::Error::Success();
    auto plan = std::make_unique<Plan>();
    plan->device = device;
    const bool fp64 = input.element_type() == ffi::F64 || input.element_type() == ffi::C128;
    plan->kind = inverse ? (fp64 ? CUFFT_Z2D : CUFFT_C2R) : (fp64 ? CUFFT_D2Z : CUFFT_R2C);
    long long length = n, input_length = inverse ? n / 2 + 1 : n;
    long long output_length = inverse ? n : n / 2 + 1;
    long long batch = input.element_count() / input_length;
    plan->copy_bytes = inverse ? input.size_bytes() : 0;
    plan->workspace_offset = (plan->copy_bytes + 255) / 256 * 256;
    auto status = cufftCreate(&plan->handle);
    if (status != CUFFT_SUCCESS) return cufft_error("cufftCreate", status);
    status = cufftSetAutoAllocation(plan->handle, 0);
    if (status != CUFFT_SUCCESS) return cufft_error("cufftSetAutoAllocation", status);
    size_t workspace_bytes = 0;
    status = cufftMakePlanMany64(plan->handle, 1, &length, &input_length, 1, input_length,
                               &output_length, 1, output_length, plan->kind, batch, &workspace_bytes);
    if (status != CUFFT_SUCCESS) return cufft_error("cufftMakePlanMany64", status);
    if (plan->workspace_offset + workspace_bytes > scratch->size_bytes())
        return ffi::Error::Internal("cuFFT workspace exceeds the supplied scratch buffer");
    cache->devices.emplace(device, std::move(plan));
    return ffi::Error::Success();
}

ffi::Error execute(cudaStream_t stream, PlanCache* cache, int32_t device, ffi::AnyBuffer input,
                   ffi::Result<ffi::AnyBuffer> output,
                   ffi::Result<ffi::Buffer<ffi::U8>> scratch,
                   int64_t /*n*/, bool /*inverse*/) {
    Plan* plan;
    {
        std::lock_guard<std::mutex> lock(cache->mutex);
        plan = cache->devices.at(device).get();
    }
    // The plan outlives cached graph replays. Serialize host-side plan mutation;
    // each execution uses its own XLA-owned scratch buffer.
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_cufft_instantiate, instantiate,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInstantiate>()
        .Attr<int64_t>("n").Attr<bool>("inverse"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_cufft_initialize, initialize,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
        .Ctx<ffi::State<PlanCache>>().Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::Buffer<ffi::U8>>()
        .Attr<int64_t>("n").Attr<bool>("inverse"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(dno_cufft_execute, execute,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::State<PlanCache>>().Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::Buffer<ffi::U8>>()
        .Attr<int64_t>("n").Attr<bool>("inverse"),
    {ffi::Traits::kCmdBufferCompatible});
