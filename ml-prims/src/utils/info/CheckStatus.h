/*
* Copyright (c) 2018, NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <string>

namespace ML
{
    namespace Utils
    {
        template<class T>
        inline bool isSuccessStatus(const T& arg);
        template<class T>
        inline std::string unSuccessReason(const T& arg);

        //===============CUBLAS RELATIVE STATUSES START===================================================
        template<>
        inline bool isSuccessStatus(const cublasStatus_t& arg) {
            return arg != CUBLAS_STATUS_SUCCESS;
        }
        template<>
        inline std::string unSuccessReason(const cublasStatus_t& arg) {
            switch (arg)
            {
                case CUBLAS_STATUS_SUCCESS : return "CUBLAS_STATUS_SUCCESS";
                case CUBLAS_STATUS_NOT_INITIALIZED : return "CUBLAS_STATUS_NOT_INITIALIZED";
                case CUBLAS_STATUS_ALLOC_FAILED : return "CUBLAS_STATUS_ALLOC_FAILED";
                case CUBLAS_STATUS_INVALID_VALUE : return "CUBLAS_STATUS_INVALID_VALUE";
                case CUBLAS_STATUS_ARCH_MISMATCH : return "CUBLAS_STATUS_ARCH_MISMATCH";
                case CUBLAS_STATUS_MAPPING_ERROR : return "CUBLAS_STATUS_MAPPING_ERROR";
                case CUBLAS_STATUS_EXECUTION_FAILED : return "CUBLAS_STATUS_EXECUTION_FAILED";
                case CUBLAS_STATUS_INTERNAL_ERROR : return "CUBLAS_STATUS_INTERNAL_ERROR";
                case CUBLAS_STATUS_NOT_SUPPORTED : return "CUBLAS_STATUS_NOT_SUPPORTED";
                case CUBLAS_STATUS_LICENSE_ERROR : return "CUBLAS_STATUS_LICENSE_ERROR";
            }
            return "Uknown";
        }
        //===============CUBLAS RELATIVE STATUSES END===================================================

        //===============CUDA RELATIVE STATUSES START===================================================
        template<>
        inline bool isSuccessStatus(const cudaError_t& arg) {
            return arg != cudaSuccess;
        }
        template<>
        inline std::string unSuccessReason(const cudaError_t& arg) {
            return std::string(cudaGetErrorString(arg));
        }
        //===============CUDA RELATIVE STATUSES END=====================================================
    }
}
