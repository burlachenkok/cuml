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
* Compiler-specific macroses
* The most part of thiswas taken from http://nadeausoftware.com/articles/2012/01/c_c_tip_how_use_compiler_predefined_macros_detect_operating_system
*/

#pragma once

#include "utils/info/HelpMacroses.h"

#if defined(__clang__)
  #define RAPIDS_COMPILER_NAME "Clang/LLVM"
  #define RAPIDS_COMPILER_VERSION_STRING_LONG __VERSION__
  #define RAPIDS_COMPILER_IS_CLANG
  #define RAPIDS_FUNCTION_NAME    __FUNCTION__
#elif defined(__ICC) || defined(__INTEL_COMPILER)
  #define RAPIDS_COMPILER_NAME "Intel ICC/ICPC"
  #define RAPIDS_COMPILER_VERSION_STRING_LONG __VERSION__
  #define RAPIDS_COMPILER_IS_INTEL
  #define RAPIDS_FUNCTION_NAME    __FUNCTION__
#elif defined(__GNUC__) || defined(__GNUG__)
  #define RAPIDS_COMPILER_NAME "GNU GCC/G++"
  #define RAPIDS_COMPILER_VERSION_STRING_LONG __VERSION__
  #define RAPIDS_COMPILER_IS_GCC
  #define RAPIDS_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(__HP_cc) || defined(__HP_aCC)
  #define RAPIDS_COMPILER_NAME "Hewlett-Packard C/aC++"
  #define RAPIDS_COMPILER_VERSION_STRING_LONG RAPIDS_STRINGIZING(__HP_aCC)
  #define RAPIDS_COMPILER_IS_HP
  #define RAPIDS_FUNCTION_NAME    __FUNCTION__
#elif defined(__IBMC__) || defined(__IBMCPP__)
  #define RAPIDS_COMPILER_NAME "IBM XL C/C++"
  #define RAPIDS_COMPILER_VERSION_STRING_LONG __xlc__
  #define RAPIDS_COMPILER_IS_IBM
  #define RAPIDS_FUNCTION_NAME    __FUNCTION__
#elif defined(_MSC_VER)
  #error "RAPIDS is not supported for building under Windows and any Windows platform compilers"
#elif defined(__PGI)
  #define RAPIDS_COMPILER_NAME "Portland Group PGCC/PGCPP"
  #define RAPIDS_COMPILER_VERSION_STRING_LONG  RAPIDS_STRINGIZING(__PGIC__) "." RAPIDS_STRINGIZING(__PGIC_MINOR) "." RAPIDS_STRINGIZING(__PGIC_PATCHLEVEL__)
  #define RAPIDS_COMPILER_IS_PGI
  #define RAPIDS_FUNCTION_NAME    __FUNCTION__
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  #define RAPIDS_COMPILER_NAME "Oracle Solaris Studio"
  #define RAPIDS_COMPILER_VERSION_STRING_LONG RAPIDS_STRINGIZING(__SUNPRO_CC)
  #define RAPIDS_COMPILER_IS_ORACLE
  #define RAPIDS_FUNCTION_NAME    __FUNCTION__
#endif
