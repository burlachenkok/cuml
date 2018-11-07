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

/** Stringizing of exprtession without evaluate it in compiler time
* @param expr expression which you would not be evaluated in compile-time and then stringinized
*/
#define RAPIDS_STRINGIZING_NO_EVAL_EXPRESSION(expr) #expr

/** Stringizing. Text which will be checked for macro expansion and arithmeitc operations during compile-time, and then stringinized
* @param expr expression which you want to evaluate in compile-time and then stringinized
*/
#define RAPIDS_STRINGIZING(expr) RAPIDS_STRINGIZING_NO_EVAL_EXPRESSION(expr)
