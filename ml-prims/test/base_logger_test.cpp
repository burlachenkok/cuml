#include <gtest/gtest.h>
#include "utils/logger/BaseLogger.h"

namespace ML {
namespace Utils {

TEST(Logging, BaseLoggerTest)
{
    using ::ML::Utils::BaseLogger;
    BaseLogger logger;
    logger << "Hel";
    logger << "lo";
    logger.Printf(" %s%c", "worl", 'd');
    EXPECT_STREQ(logger.getBufferedMessage().c_str(), "Hello world");

    logger.flush();
    EXPECT_TRUE(logger.getBufferedMessage().empty());

    logger << "test" << -123;
    EXPECT_STREQ(logger.getBufferedMessage().c_str(), "test-123");

    logger << "test" << -123 << "\n";
    EXPECT_TRUE(logger.getBufferedMessage().empty());

}

} // end namespace Utils
} // end namespace ML
