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
* Rather complicated logger with different message types, and targets for send messages
*/

#pragma once

#include "utils/logger/BaseLogger.h"
#include "utils/info/SystemInfo.h"

#include <stdint.h>
#include <stdlib.h>

#include <vector>
#include <string>
#include <functional>

namespace ML
{
    namespace Utils
    {
        /** Logger which is needed for track errors/warning and also usefull by itself
        */
        class Logger : public BaseLogger
        {
        public:

            /** Message type for logging
            */
            enum LogMessageType
            {
                eLogDebug   = 0x1 << 0,  ///< debug messages
                eLogWarning = 0x1 << 1,  ///< warn messages
                eLogError   = 0x1 << 2,  ///< error message
                eLogAll = LogMessageType::eLogDebug | LogMessageType::eLogWarning | LogMessageType::eLogError ///< all message types
            };

            /** Print policy for logging.
            *  - with what information log message should be augmented
            *  - what is target device for logging
            */
            enum PrintPolicy
            {
                eDefault   = 0,                ///< Initialize with nothing or default print policy
                eStdOutput = 0x1 << 0,         ///< Output to stdout
                eStdErrOutput = 0x1 << 1,      ///< Output to stderr
                eFileOutput = 0x1 << 2,        ///< Output print to specific file

                ePrintThreadId = 0x1 << 3,     ///< Extra print before message - thread id
                ePrintProcessId = 0x1 << 4,    ///< Extra print before message - process id
                ePrintGpuId      = 0x1 << 5,   ///< Extra print before message - current active GPU

                ePrintLocalTime = 0x1 << 6,    ///< Extra print before message - print local time
                ePrintFunctionName = 0x1 << 7, ///< Extra print before message - function name
                ePrintFileName = 0x1 << 8,     ///< Extra print before message - file name
                ePrintLineNumber = 0x1 << 9,   ///< Extra print before message - line number
                ePrintAllExtra = PrintPolicy::ePrintThreadId | PrintPolicy::ePrintProcessId | PrintPolicy::ePrintGpuId |
                                 PrintPolicy::ePrintLocalTime | PrintPolicy::ePrintFunctionName | PrintPolicy::ePrintFileName | PrintPolicy::ePrintLineNumber, ///< Extra print all possible information
                eTerminateOnError = 0x1 << 10, ///< Terminate process if error occured
            };

            /* Logger ctor. Setup logger to log messages with severity in [0, severity]
            * @param thePolicy where to print and what to print
            * @param theAcceptMessages accept messages filter
            * @param theDumpFileName temporary string with filename if (policy & PrintPolicy::eFileOutput)
            */
            Logger(uint32_t thePolicy = PrintPolicy::eStdErrOutput | PrintPolicy::ePrintAllExtra,
                   uint32_t theAcceptMessages = LogMessageType::eLogAll)
            :  acceptableMessages(theAcceptMessages)
            ,  printPolicy(thePolicy)
            ,  lastMessageInfo()
            {}

            /** Specialized initializationwith the env.variable which configure behavior of logging mechanism
            * @return prepared logger initialized with help from setuped environment variables
            * @remark to have support of this function environment should support concept of env.variables
            * @remark env.variable RAPIDS_LOG_DEST should be setuped into "stdout" or "stderr" or "some filename" or "should not be setuped at all"
            * @remark env.variable RAPIDS_LOG_MESSAGES should be setuped into string which can containt "warning", "error", "debug", "all" as substrings
            */
            static Logger initializeWithEnvironment()
            {
                Logger logger = Logger(PrintPolicy::eDefault, LogMessageType::eLogAll);
                if (isEnvVariableExist("RAPIDS_LOG_DEST"))
                {
                    std::string logDest = envVariableValue("RAPIDS_LOG_DEST");
                    if (logDest == "stdout")
                    {
                        logger.printPolicy |= PrintPolicy::eStdOutput;
                    }
                    else if (logDest == "stderr")
                    {
                        logger.printPolicy |= PrintPolicy::eStdErrOutput;
                    }
                    else
                    {
                        logger.printPolicy |= PrintPolicy::eFileOutput;
                        logger.dumpFnames.push_back(logDest);
                    }
                }

                if (isEnvVariableExist("RAPIDS_LOG_MESSAGES"))
                {
                    std::string logMessages = envVariableValue("RAPIDS_LOG_MESSAGES");
                    if (logMessages.find("debug") != std::string::npos)
                        logger.acceptableMessages |= LogMessageType::eLogDebug;
                    if (logMessages.find("warning") != std::string::npos)
                        logger.acceptableMessages |= LogMessageType::eLogWarning;
                    if (logMessages.find("error") != std::string::npos)
                        logger.acceptableMessages |= LogMessageType::eLogError;
                    if (logMessages.find("all") != std::string::npos)
                        logger.acceptableMessages |= LogMessageType::eLogAll;
                }

                // Print all available extra information
                logger.printPolicy |= PrintPolicy::ePrintAllExtra;

                return logger;
            }

            /** Append print log to file
            * @param theDumpFileName target filename in which logging will be appended
            * @return reference to *this
            */
            Logger& appendPrintLogToFile(const std::string& theDumpFileName)
            {
                size_t sz = dumpFnames.size();
                for (size_t i = 0; i < sz; ++i)
                {
                    if (dumpFnames[i] == theDumpFileName)
                        return *this;
                }

                dumpFnames.push_back(theDumpFileName);
                return *this;
            }

            /** Logger dtor
            */
            virtual ~Logger() {}

            /** Receive number of how much errors messages was emmmited
            * @return invoke count
            */
            uint32_t getErrorMessagesCount() const
            {
                return totalMessageCounters.errors;
            }

            /** Receive number of how much warnings messages was emmmited
            * @return invoke count
            */
            uint32_t getWarningMessagesCount() const
            {
                return totalMessageCounters.warnings;
            }

            /** Receive number of how much debug messages was emmmited
            * @return invoke count
            */
            uint32_t getDebugMessagesCount() const
            {
                return totalMessageCounters.debug;
            }

            typedef void(CallBackNextLogMessage)(uint32_t nextMsgType, const char* fileName, const char* functionName, int lineNumber);
            void registerCallBackForNextLogMessage(std::function<CallBackNextLogMessage> callback)
            {
                updateCallBack = callback;
            }

            //===========================================================================================================================//
            /** Call of this function should be done from source code contain information to log
            * @param nextMsgType next message type
            * @param fileName current source file name
            * @param functionName current function name
            * @param lineNumber current source file line number
            * @return *this
            * @remark thif function update info about last message, update counters, and call user-defined callback function
            */
            virtual BaseLogger& updateNextMessage(uint32_t nextMsgType, const char* fileName, const char* functionName, int lineNumber)
            {
                if (lastMessageInfo.msgType != nextMsgType && lastMessageInfo.isOk())
                {
                    // if message has different type then force flushing anything relative to previous message
                    flush();
                }

                lastMessageInfo.msgType = nextMsgType;
                lastMessageInfo.fileName = fileName;
                lastMessageInfo.functionName = functionName;
                lastMessageInfo.lineNumber = lineNumber;

                if (nextMsgType & LogMessageType::eLogDebug)
                    totalMessageCounters.debug++;
                if (nextMsgType & LogMessageType::eLogWarning)
                    totalMessageCounters.warnings++;
                if (nextMsgType & LogMessageType::eLogError)
                    totalMessageCounters.errors++;
                if (updateCallBack) {
                    updateCallBack(nextMsgType, fileName, functionName, lineNumber);
                }

                return *this;
            }
            //===========================================================================================================================//

        protected:
            /** Print output. Default implementation.
            * @param fullMsg full message that should be logged
            * @remark there is explicit syncronization with stdout/stderr and dumped file. It can be considered in the same time as plus and minus
            */
            virtual void doPrintOutput(const std::string& fullMsg)
            {
                if (printPolicy & PrintPolicy::eStdOutput) {
                    fputs(fullMsg.c_str(), stdout);
                    fflush(stdout);
                }

                if (printPolicy & PrintPolicy::eStdErrOutput) {
                    fputs(fullMsg.c_str(), stderr);
                    fflush(stderr);
                }

                if (printPolicy & PrintPolicy::eFileOutput) {
                    for (const std::string& fname : dumpFnames)
                    {
                        FILE * file = fopen (fname.c_str(), "at");
                        if (file)
                        {
                            fputs(fullMsg.c_str(), file);
                            fclose(file);
                        }
                    }
                }
            }

            /** Create full log message, notify doPrintOutput about it and clear internal buffer
            */
            virtual void doFlushManipProcess()
            {
                if (lastMessageInfo.fileName == 0)
                    assert(!"Please use ML_LOG macros to dump log messages");

                std::string printMessage = getBufferedMessage();
                std::stringstream extraPrinting;

                if ((lastMessageInfo.msgType & acceptableMessages) == 0)
                {
                    // not acceptable message
                    return;
                }

                if (printMessage.empty())
                {
                    // nothing to print
                    return;
                }

                if ((printPolicy & PrintPolicy::ePrintAllExtra) != 0) // Check that at least there are something to print
                {
                    int print_info_item_count = 0;
                    if (printPolicy & PrintPolicy::ePrintProcessId)
                    {
                        if (print_info_item_count++ > 0)
                            extraPrinting << "/";
                        extraPrinting << "PROCESS:" << currentProcessId();
                    }
                    if (printPolicy & PrintPolicy::ePrintThreadId)
                    {
                        if (print_info_item_count++ > 0)
                            extraPrinting << "/";
                        extraPrinting << "THREAD:" << currentThreadId();
                    }
                    if (printPolicy & PrintPolicy::ePrintGpuId)
                    {
                        if (print_info_item_count++ > 0)
                            extraPrinting << "/";
                        extraPrinting << "GPU:" << currentGPU();
                    }

                    if (printPolicy & PrintPolicy::ePrintLocalTime)
                    {
                        if (print_info_item_count++ > 0)
                            extraPrinting << "/";
                        extraPrinting << currentLocalTime();
                    }
                    if (printPolicy & PrintPolicy::ePrintFileName)
                    {
                        if (print_info_item_count++ > 0)
                            extraPrinting << "/";
                        extraPrinting << "file: " << lastMessageInfo.fileName;
                    }
                    if (printPolicy & PrintPolicy::ePrintLineNumber)
                    {
                        if (print_info_item_count++ > 0)
                            extraPrinting << "/";
                        extraPrinting << "line: " << lastMessageInfo.lineNumber;
                    }
                    if (printPolicy & PrintPolicy::ePrintFunctionName)
                    {
                        if (print_info_item_count++ > 0)
                            extraPrinting << "/";
                        extraPrinting << "func: " << lastMessageInfo.functionName;
                    }
                }
                std::string fullMsg;
                if (extraPrinting.str().empty())
                {
                    fullMsg = printMessage;
                }
                else
                {
                    fullMsg = std::string("{") + extraPrinting.str() + std::string("}") + printMessage;
                }

                doPrintOutput(fullMsg);
                if ((printPolicy & PrintPolicy::eTerminateOnError) && (lastMessageInfo.msgType & LogMessageType::eLogError))
                {
                    // Terminate. Bad from software point of view, but good for debug.
                    exit(1);
                }
            }

        protected:

            /** Class to handle simple statistics relative to errors/warnings
            */
            struct Counters
            {
                Counters()
                : debug(0)
                , errors(0)
                , warnings(0)
                {}

                uint32_t debug;                    ///< number of debug messages
                uint32_t errors;                   ///< number of errors
                uint32_t warnings;                 ///< number of warning
            };
            Counters totalMessageCounters;         ///< total message counter
            uint32_t acceptableMessages;           ///< where and what to print
            uint32_t printPolicy;                  ///< where and what to print
            std::vector<std::string> dumpFnames;   ///< filename if policy & PrintPolicy::eFileOutput
            std::function<CallBackNextLogMessage> updateCallBack; ///< extra callback for update messages (originally to create extra logic for google test)

            struct MessageInfo
            {
                uint32_t msgType;         ///< last message type which have been pushed
                const char* fileName;     ///< source filename from which message have been passed
                const char* functionName; ///< function name where message occured
                int lineNumber;           ///< source file line number

                bool isOk() const {
                    return fileName != nullptr && functionName != nullptr;
                }
            };
            MessageInfo lastMessageInfo;  ///< Place where is stored updateNextMessage info
        };
    }
}
