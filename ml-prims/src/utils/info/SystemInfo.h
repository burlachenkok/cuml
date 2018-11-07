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

#include <assert.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>

#include <sstream>
#include <float.h>
#include <time.h>
#include <string>
#include <stdint.h>

#include <cuda_runtime_api.h>

namespace ML
{
    namespace Utils
    {
        /** Get current number of active GPU which will receive all CUDA commands from the host
        * @return current gpu id
        * @remark Any device memory allocated from the host thread will be physically resident on that device
        * @remark Any host memory allocated with CUDA runtime functions will have its lifetime associated with that device
        * @remark Any streams or events created from the host thread will be associated with that device
        * @remark Any kernels launched from the host thread will be executed on that device
        */
        inline int currentGPU()
        {
            int device = 0;
            cudaError_t err = cudaGetDevice(&device);
            (void)err;
            assert(err == cudaSuccess);
            return device;
        }

        /** Get current process id
        * @return current process id
        */
        inline uint32_t currentProcessId() {
            assert(sizeof(getpid()) <= sizeof(uint32_t));
            return (uint32_t)getpid();
        }

        /** Get current thread id
        * @return current thread id
        */
        inline uint64_t currentThreadId() {
            assert(sizeof(pthread_self()) <= sizeof(uint64_t));
            return (uint64_t)pthread_self();
        }

        /** Is envrinoment variable exist
        * @return flag that envrinoment variable exist
        */
        inline bool isEnvVariableExist(const char* envVariableName)
        {
            char* envVar = getenv(envVariableName);
            if (!envVar)
                return false;
            else
                return true;
        }

        /** Get envrinoment variable by name
        * @param envVariableName name of envrinoment variable
        * @return true if envrinoment variable have been setuped
        */
        inline std::string envVariableValue(const char* envVariableName)
        {
            char* envVar = getenv(envVariableName);
            if (!envVar)
                return std::string();
            else
                return std::string(envVar);
        }

        /** Get string represented a time in some it's own defined (but clean) format
        * @return string represented current local time up to second
        * @todo probably need more details information about current microseconds
        */
        inline std::string currentLocalTime()
        {
            time_t t = time(NULL);
            tm * tg = localtime(&t);
            std::stringstream s;
            s << "(" << tg->tm_hour << ":" << tg->tm_min << ":" << tg->tm_sec;
            return s.str();
        }
    }
}
