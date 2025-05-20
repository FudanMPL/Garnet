
#include "Math/gfp.hpp"
#include "GC/TinierSecret.h"
#include "Processor/FieldMachine.h"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    DishonestMajorityFieldMachine<Share>(argc, argv, opt);
}
