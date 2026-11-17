/**
 *  @file test/header_core.cpp
 *  @author Ash Vardanian
 *  @date November 5, 2026
 *  @brief The dependency-free core's proof: the basic RAM index compiles with no sibling library
 *      on the include path.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include "usearch/basic/ram_index.hpp"
