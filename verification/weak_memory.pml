/**
 *  @brief The memory model, kept in ForkUnion beside the atomics it describes; every model
 *      here includes it by this name, as ForkUnion's own models do.
 */
#if __has_include("../forkunion/verification/weak_memory.pml")
#include "../forkunion/verification/weak_memory.pml"
#else
#include <weak_memory.pml> // a ForkUnion checkout's `verification/` on the include path, as `check.sh` sets it
#endif
