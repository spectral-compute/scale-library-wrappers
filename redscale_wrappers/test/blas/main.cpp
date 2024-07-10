#include <cuda.h>
#ifdef __REDSCALE__
#include <redscale.h>
#endif // __REDSCALE__
#include <gtest/gtest.h>

const char *argv0 = nullptr;

int main(int argc, char **argv)
{
#ifdef __REDSCALE__
    redscale::Exception::enable();
#endif // __REDSCALE__
    argv0 = argv[0];
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
