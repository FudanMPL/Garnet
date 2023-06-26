#include "SPDZ.hpp"

#include "Protocols/MascotPrep.hpp"
#include "Processor/FieldMachine.hpp"
#include "Math/gfp.hpp"

template class FieldMachine<Share, Share, DishonestMajorityMachine>;
