/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

#include "CudaRt.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
	std::cout << "frontend CudaRt_cudaFree !!!!!!!" << std::endl;
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::Execute("cudaFree");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray *array) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments((void *) array);
    CudaRtFrontend::Execute("cudaFreeArray");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    CudaRtFrontend::Prepare();
    // Achtung: skip adding devPtr
    CudaRtFrontend::AddSymbolForArguments(symbol);
    CudaRtFrontend::Execute("cudaGetSymbolAddress");
    if (CudaRtFrontend::Success())
        *devPtr = CudaUtil::UnmarshalPointer(CudaRtFrontend::GetOutputString());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(size);
    CudaRtFrontend::AddSymbolForArguments(symbol);
    CudaRtFrontend::Execute("cudaGetSymbolSize");
    if (CudaRtFrontend::Success())
        *size = *(CudaRtFrontend::GetOutputHostPointer<size_t > ());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags) {
    // Achtung: we can't use mapped memory
    return cudaErrorMemoryAllocation;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    // Achtung: falling back to the simplest method because we can't map memory
#ifndef CUDA_VERSION
#error CUDA_VERSION not defined
#endif
#if CUDA_VERSION >= 2030
    *pFlags = cudaHostAllocDefault;
#endif
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::Execute("cudaMalloc");

    if (CudaRtFrontend::Success())
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
        cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMalloc3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

#if CUDART_VERSION >= 3000
extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, cudaExtent extent,
        unsigned int flags) {
#else
extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, cudaExtent extent) {
#endif
    // FIXME: implement
    cerr << "*** Error: cudaMalloc3DArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

// FIXME: new mapping way

#if CUDART_VERSION >= 3000
extern "C" __host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags) {
#else
extern "C" __host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height) {
#endif
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddHostPointerForArguments(desc);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMallocArray");
    if (CudaRtFrontend::Success())
        *arrayPtr = (cudaArray *) CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, 
        size_t *pitch, size_t width, size_t height) {
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddVariableForArguments(*pitch);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMallocPitch");

    if (CudaRtFrontend::Success()) {
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();
        *pitch = CudaRtFrontend::GetOutputVariable<size_t>();
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DArrayToArray() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DFromArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DFromArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DToArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const cudaMemcpy3DParms *p) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy3DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t count,
        cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyArrayToArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind, cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddVariableForArguments(kind);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
            CudaRtFrontend::AddVariableForArguments(stream);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
            CudaRtFrontend::AddVariableForArguments(stream);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
            CudaRtFrontend::AddVariableForArguments(stream);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
            CudaRtFrontend::AddVariableForArguments(stream);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromArrayAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol,
        size_t count, size_t offset,
        cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToHost:
            // Achtung: adding a fake host pointer 
            CudaRtFrontend::AddDevicePointerForArguments((void *) 0x666);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
            CudaRtFrontend::AddStringForArguments((char*)symbol);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
            CudaRtFrontend::AddStringForArguments((char*)symbol);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol");
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromSymbolAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *) dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToArray");
            break;
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *) dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToArray");
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src,
        size_t count, size_t offset,
        cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
            CudaRtFrontend::AddStringForArguments((char*)symbol);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol");
            break;
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
            CudaRtFrontend::AddStringForArguments((char*)symbol);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol");
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToSymbolAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int c, size_t count) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(c);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::Execute("cudaMemset");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset2D(void *mem, size_t pitch, int c, size_t width,
        size_t height) {
    // FIXME: implement
    cerr << "*** Error: cudaMemset2D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value,
        cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMemset3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}
