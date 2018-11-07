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

/** @file
* Macroses for handle specific type of warnings/errors
*/

#pragma once

#include "utils/logger/Logger.h"
#include "utils/info/CompilerInfoMacroses.h"
#include "utils/info/ArchMacroses.h"
#include "utils/info/CheckStatus.h"

#define ML_LOG_ERROR(logger)   (logger).updateNextMessage(::ML::Utils::Logger::LogMessageType::eLogError,\
                                                                       __FILE__,\
                                                                       RAPIDS_FUNCTION_NAME,\
                                                                       __LINE__)

#define ML_LOG_WARNING(logger) (logger).updateNextMessage(::ML::Utils::Logger::LogMessageType::eLogWarning,\
                                                                         __FILE__,\
                                                                         RAPIDS_FUNCTION_NAME,\
                                                                         __LINE__)

#define ML_LOG_DEBUG(logger)    (logger).updateNextMessage(::ML::Utils::Logger::LogMessageType::eLogDebug,\
                                                                         __FILE__,\
                                                                         RAPIDS_FUNCTION_NAME,\
                                                                         __LINE__)

#define ML_EVAL_EXPRESSION_AND_CHECK(logger, expr) \
    do {\
        auto status = (expr); \
        if (!::ML::Utils::isSuccessStatus(status)) { \
            ML_LOG_ERROR(logger) << " Unsuccessful opertation occured in " << __FILE__ << ":" << __LINE__ << "." \
                                 << " Reason: " <<  ::ML::Utils::unSuccessReason(status) << "\n"; \
        }\
    }\
    while (0)

#define ML_LOG_TOOLCHAIN_INFO(logger) \
    do { ML_LOG_DEBUG(logger) << "Compiler: " << RAPIDS_COMPILER_NAME << "/" << RAPIDS_COMPILER_VERSION_STRING_LONG << "\n"; } while(0)

#define ML_SYSTEM_INFO(logger) \
    do { ML_LOG_DEBUG(logger) << "Architecture: " << RAPIDS_ARCH_NAME << "\n"; } while(0)
