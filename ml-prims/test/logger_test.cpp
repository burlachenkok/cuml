#include <gtest/gtest.h>
#include "utils/logger/Logger.h"
#include "utils/logger/LoggerMacroses.h"

#include <stdint.h>
#include <string>

namespace
{
    using namespace ::ML::Utils;

    class TestLogger : public Logger
    {
    public:
        TestLogger(uint32_t thePolicy = Logger::PrintPolicy::eStdErrOutput | Logger::PrintPolicy::ePrintAllExtra, uint32_t theAcceptMessages = Logger::LogMessageType::eLogAll)
        : Logger(thePolicy, theAcceptMessages)
        {
        }

        void doPrintOutput(const std::string& fullMsg) override
        {
            lastPrintOutput = fullMsg;
        }

        std::string lastPrintOutput;
    };
}

namespace ML {
namespace Utils {

TEST(Logging, LoggerPrintingTest)
{
    using ML::Utils::Logger;
    {
        TestLogger logger2( Logger::PrintPolicy::eDefault,Logger::LogMessageType::eLogError|Logger::LogMessageType::eLogWarning);

        ML_LOG_WARNING(logger2) << "Hello Warn\n";
        EXPECT_TRUE(logger2.getBufferedMessage().empty());
        EXPECT_TRUE(logger2.lastPrintOutput.find("Hello Warn\n") == 0);

        ML_LOG_ERROR(logger2) << "Hello Error";
        EXPECT_STREQ(logger2.getBufferedMessage().c_str(), "Hello Error");
        logger2.Printf("%c", '\n');
        EXPECT_STREQ(logger2.getBufferedMessage().c_str(), "");
        EXPECT_STREQ(logger2.lastPrintOutput.c_str(), "Hello Error\n");
    }
    {
        TestLogger logger3( Logger::PrintPolicy::ePrintLineNumber| Logger::PrintPolicy::ePrintFileName,
                            Logger::LogMessageType::eLogWarning );

        ML_LOG_WARNING(logger3) << "Hello Warn" << "\n";
        EXPECT_TRUE(logger3.getBufferedMessage().empty());
        EXPECT_TRUE(logger3.lastPrintOutput.find("Hello Warn\n") > 0);
        EXPECT_TRUE(logger3.lastPrintOutput.find("line: ") != std::string::npos);
        EXPECT_TRUE(logger3.lastPrintOutput.find("file: ") != std::string::npos);
        EXPECT_TRUE(logger3.lastPrintOutput.find("logger_test.cpp") != std::string::npos);

        ML_LOG_WARNING(logger3) << "Hello Warn2\n";
        EXPECT_TRUE(logger3.getBufferedMessage().empty());
        EXPECT_TRUE(logger3.lastPrintOutput.find("Hello Warn2\n") != std::string::npos);
        EXPECT_TRUE(logger3.lastPrintOutput.find("Hello Warn\n") == std::string::npos);
        logger3.flush();
        logger3.lastPrintOutput = std::string();

        ML_LOG_ERROR(logger3) << "Error should be pushed due to policy\n";
        EXPECT_TRUE(logger3.lastPrintOutput.empty());
        ML_LOG_WARNING(logger3) << "Warning should be pushed due to policy\n";
        EXPECT_FALSE(logger3.lastPrintOutput.empty());
    }
}

} // end namespace Utils
} // end namespace ML
