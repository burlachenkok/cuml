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
* Architecture specific macroses
* Information to make I took from http://nadeausoftware.com/articles/2012/01/c_c_tip_how_use_compiler_predefined_macros_detect_operating_system
*/

#pragma once

#if defined(__ia64) || defined(__itanium__) || defined(_M_IA64)
  #define RAPIDS_ARCH_NAME "Itanium"
  #define RAPIDS_ARCH_ITANIUM 1
  #define RAPIDS_ARCH_LITTLE_ENDIAN 1
  #define RAPIDS_ARCH_BIG_ENDIAN    0

#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
  #if defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__) || defined(__64BIT__) || defined(_LP64) || defined(__LP64__)
    #define RAPIDS_ARCH_NAME "Power pc 64"
    #define RAPIDS_ARCH_POWER_PC_64 1
  #else
    #define RAPIDS_ARCH_NAME "Power pc 32"
    #define RAPIDS_ARCH_POWER_PC_32 1
  #endif
#elif defined(__sparc)
  #define RAPIDS_ARCH_NAME "Sparc"
  #define RAPIDS_ARCH_SPARC 1
  #define RAPIDS_ARCH_LITTLE_ENDIAN 0
  #define RAPIDS_ARCH_BIG_ENDIAN 1
#elif defined(__x86_64__) || defined(_M_X64)
  #define RAPIDS_ARCH_NAME "AMD, Intel x86 64 bit"
  #define RAPIDS_ARCH_X86_64BIT 1
  #define RAPIDS_ARCH_LITTLE_ENDIAN 1
  #define RAPIDS_ARCH_BIG_ENDIAN    0
#elif defined(__i386) || defined(_M_IX86)
  #define RAPIDS_ARCH_NAME "AMD, Intel x86 32 bit"
  #define RAPIDS_ARCH_X86_32BIT 1
  #define RAPIDS_ARCH_LITTLE_ENDIAN 1
  #define RAPIDS_ARCH_BIG_ENDIAN    0
#endif
