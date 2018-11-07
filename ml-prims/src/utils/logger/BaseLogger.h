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
* Base logger implementation
*/

#pragma once

#include <sstream>
#include <stdarg.h>
#include <string.h>

namespace ML
{
    namespace Utils
    {
        /** Base loger class
        */
        class BaseLogger {
        public:
            BaseLogger() {
            }

            BaseLogger(const BaseLogger& rhs) {
                str_.str(rhs.str_.str());
            }

            /** Push arg to std string stream
            * @param arg something what can be passed to std::stringstream
            * @return *this
            */
            template<class T>
            BaseLogger& operator << (const T& arg) {
                str_ << arg;
                return *this;
            }

            /** Push arg whcih is ASCII-Z string to internal std string stream
            * @param arg something what can be passed to std::stringstream
            * @return *this
            * @remark if string contains new line character then there special function perform-end-of-manipulator is called
            */
            BaseLogger& operator << (const char* arg) {
                str_ << arg;
                if (isStrContain(arg, '\n'))
                    doEolManipProcess();
                return *this;
            }

            /** Push arg whcih is ASCII-Z string to internal std string stream
            * @param arg something what can be passed to std::stringstream
            * @return *this
            * @remark if string contains new line character then there special function perform-end-of-manipulator is called
            */
            BaseLogger& operator << (char* arg) {
                return *this << static_cast<const char*>(arg);
            }

            /** Push arg whcih is std::string to internal std string stream
            * @param arg something what can be passed to std::stringstream
            * @return *this
            * @remark if string contains new line character then there special function perform-end-of-manipulator is called
            */
            BaseLogger& operator << (const std::string& arg) {
                return (*this << (arg.c_str()));
            }

            /** C-style printf function for log activity
            * @param str prinf c format string
            * @return *this
            */
            BaseLogger& Printf(const char * str, ...) {
                size_t buf_size = 512;
                char* buf = new char[buf_size];

                va_list args;
                va_start(args, str);
                for (;;)
                {
                    int buf_elements = vsprintf(buf, str, args);
                    if (size_t(buf_elements) < buf_size)
                    {
                        break;
                    }
                    else
                    {
                        // Buffer is not enough relocate buffer
                        buf_size *= 2;
                        delete[] buf;
                        buf = new char[buf_size];
                    }
                }
                va_end(args);

                *this << buf;

                delete[]buf;
                return *this;
            }

            /** Get currently buffered logged message
            * @return copy of buffered message
            * @remark if it is empty it can mean that nobody push messages or messages have been flushed to the target
            */
            std::string getBufferedMessage() const
            {
                return str_.str();
            }

            /** Cleanup internal storage
            */
            void cleanBufferedMessage() {
                str_.str(std::string());
            }

            /** Perform flush of buffer to target place and cleanup accumualted buffer
            * @return reference to itself
            */
            BaseLogger& flush()
            {
                doFlushManipProcess();
                cleanBufferedMessage();
                return *this;
            }

        protected:

            /** Helper function for find symbol in string
            * @param str ASCII-Z string in which searching is occur
            * @param symbol2search symbol which is searched in the string
            * @return true if string contains symbol2search
            * @todo place into some other place
            */
            static bool isStrContain(const char* str, char symbol2search)
            {
                size_t len = strlen(str);
                for (size_t i = 0; i < len; ++i)
                {
                    if (str[i] == symbol2search)
                    {
                        return true;
                    }
                }
                return false;
            }

            /** Flush internal buffer so somewhere and clean them
            */
            virtual void doFlushManipProcess() {
            }

            /** Perform custom logic when end of line received. Be default call (probably rederived) doFlushManipProcess
            */
            void doEolManipProcess() {
                flush();
            }

            std::stringstream str_; ///< used internal object which accumulated logger input
        };
    }
}
