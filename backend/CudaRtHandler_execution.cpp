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

#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(ConfigureCall) {
    /* cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
     * size_t sharedMem, cudaStream_t stream) */
    dim3 gridDim = input_buffer->Get<dim3>();
    dim3 blockDim = input_buffer->Get<dim3>();
    size_t sharedMem = input_buffer->Get<size_t>();
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();

    cudaError_t exit_code = cudaConfigureCall(gridDim, blockDim, sharedMem,
            stream);

    return new Result(exit_code);
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
CUDA_ROUTINE_HANDLER(FuncGetAttributes) {
    cudaFuncAttributes *guestAttr = input_buffer->Assign<cudaFuncAttributes>();
    char *handler = input_buffer->AssignString();
    const char *entry = pThis->GetDeviceFunction(handler);
    Buffer * out = new Buffer();
    cudaFuncAttributes *attr = out->Delegate<cudaFuncAttributes>();
    memmove(attr, guestAttr, sizeof(cudaFuncAttributes));
    
    cudaError_t exit_code = cudaFuncGetAttributes(attr, entry);

    return new Result(exit_code, out);
}
#endif

CUDA_ROUTINE_HANDLER(Launch) {
//    int ctrl;
//
//    CUcontext pctx;
//
//    CUresult exit_code_cu = cuCtxPopCurrent(&pctx);
//
//    // Create a context for this thread if one does not already exist.
//    if (exit_code_cu == CUDA_ERROR_INVALID_CONTEXT)
//    {
//    	std::cout << "cudaFree: creating new context" << std::endl;
//    	exit_code_cu = cuCtxCreate (&pctx, 0, 0);
//    }
//
//    if (exit_code_cu == CUDA_SUCCESS)
//    {
//		// cudaConfigureCall
//		ctrl = input_buffer->Get<int>();
//		if(ctrl != 0x434e34c)
//			throw "Expecting cudaConfigureCall";
//
//		dim3 gridDim = input_buffer->Get<dim3>();
//		dim3 blockDim = input_buffer->Get<dim3>();
//		size_t sharedMem = input_buffer->Get<size_t>();
//		cudaStream_t stream = input_buffer->Get<cudaStream_t>();
//
//		std::cout << "cudaLaunch: cudaConfigureCall" << std::endl;
////		cudaError_t exit_code = cudaConfigureCall(gridDim, blockDim, sharedMem,
////				stream);
////
////		if(exit_code != cudaSuccess)
////			return new Result(exit_code);
//
//		// cudaSetupArgument
//		char argBuffer[256];
//		int bufferSize = 0;
//		while((ctrl = input_buffer->Get<int>()) == 0x53544147) {
//			void *arg = input_buffer->AssignAll<char>();
//			size_t size = input_buffer->Get<size_t>();
//			size_t offset = input_buffer->Get<size_t>();
//
//			memcpy(&argBuffer[offset], arg, size);
//			bufferSize += size;
//
//			std::cout << "cudaLaunch: cudaSetupArgument with size " << size << " at offset " << offset << std::endl;
//	//        exit_code = cudaSetupArgument(arg, size, offset);
//	//        if(exit_code != cudaSuccess)
//	//            return new Result(exit_code);
//		}
//		std::cout << "cudaLaunch: cudaSetupArgBuffer with size " << bufferSize << std::endl;
//	    void *kernel_launch_config[5] =
//	    {
//	        CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
//	        CU_LAUNCH_PARAM_BUFFER_SIZE,    &bufferSize,
//	        CU_LAUNCH_PARAM_END
//	    };
//		// cudaLaunch
//		if(ctrl != 0x4c41554e)
//			throw "Expecting cudaLaunch";
//
//		char *handler = input_buffer->AssignString();
//		const char *entry = pThis->GetDeviceFunction(handler);
//
//		CUmodule module;
//		CUfunction function;
//		for (std::map<std::string, void **>::iterator i = pThis->mpFatBinary->begin();
//				i != pThis->mpFatBinary->end(); ++i)
//		{
//			std::cout << "handler: " << i->first << "; cubin: " << i->second << std::endl;
//			void **fatCubinHandle = (void**)(i->second);
//			exit_code_cu = cuModuleLoadFatBinary (&module, &fatCubinHandle);
//			std::cout << "cuModuleLoadFatBinary code: " << exit_code_cu << std::endl;
//			exit_code_cu = cuModuleGetFunction(&function, module, entry);
//			std::cout << "cuModuleGetFunction code: " << exit_code_cu << std::endl;
//			exit_code_cu = cuLaunchKernel(function,
//					gridDim.x, gridDim.y, gridDim.z,
//					blockDim.x, blockDim.y, blockDim.z,
//					sharedMem, stream, NULL, kernel_launch_config);
//			std::cout << "cuLaunchKernel code: " << exit_code_cu << std::endl;
//		}
//		cudaError_t exit_code = cudaLaunchKernel(entry,
//				gridDim, blockDim, (void **)argBuffer, sharedMem, stream);
//		std::cout << "cudaLaunch: cudaLaunch " << entry << std::endl;
//
//		std::cout << "cudaLaunch: cudaLaunch complete" << std::endl;
//		return new Result(exit_code);
//    }
//    return new Result (CUDA_ERROR_UNKNOWN);
    int ctrl;

    // cudaConfigureCall
    ctrl = input_buffer->Get<int>();
    if(ctrl != 0x434e34c)
        throw "Expecting cudaConfigureCall";

    dim3 gridDim = input_buffer->Get<dim3>();
    dim3 blockDim = input_buffer->Get<dim3>();
    size_t sharedMem = input_buffer->Get<size_t>();
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();

    std::cout << "cudaLaunch: cudaConfigureCall" << std::endl;
    cudaError_t exit_code = cudaConfigureCall(gridDim, blockDim, sharedMem,
            stream);

    if(exit_code != cudaSuccess)
        return new Result(exit_code);

    // cudaSetupArgument
    while((ctrl = input_buffer->Get<int>()) == 0x53544147) {
        void *arg = input_buffer->AssignAll<char>();
        size_t size = input_buffer->Get<size_t>();
        size_t offset = input_buffer->Get<size_t>();

        std::cout << "cudaLaunch: cudaSetupArgument" << std::endl;
        exit_code = cudaSetupArgument(arg, size, offset);
        if(exit_code != cudaSuccess)
            return new Result(exit_code);
    }

    // cudaLaunch
    if(ctrl != 0x4c41554e)
        throw "Expecting cudaLaunch";

    char *handler = input_buffer->AssignString();
    const char *entry = pThis->GetDeviceFunction(handler);
    std::cout << "cudaLaunch: cudaLaunch" << std::endl;
    exit_code = cudaLaunch(entry);

    std::cout << "cudaLaunch: cudaLaunch complete" << std::endl;
    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(SetDoubleForDevice) {
    double *guestD = input_buffer->Assign<double>();
    Buffer *out = new Buffer();
    double *d = out->Delegate<double>();
    memmove(d, guestD, sizeof(double));

    cudaError_t exit_code = cudaSetDoubleForDevice(d);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(SetDoubleForHost) {
    double *guestD = input_buffer->Assign<double>();
    Buffer *out = new Buffer();
    double *d = out->Delegate<double>();
    memmove(d, guestD, sizeof(double));

    cudaError_t exit_code = cudaSetDoubleForHost(d);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(SetupArgument) {
    /* cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) */
    size_t offset = input_buffer->BackGet<size_t>();
    size_t size = input_buffer->BackGet<size_t>();
    void *arg = input_buffer->Assign<char>(size);

    cudaError_t exit_code = cudaSetupArgument(arg, size, offset);

    return new Result(exit_code);
}
