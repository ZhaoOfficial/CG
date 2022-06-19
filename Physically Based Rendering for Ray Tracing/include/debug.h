#ifndef _PBRT_DEBUG_H_
#define _PBRT_DEBUG_H_

#include "common.h"

PBRT_NAMESPACE_START

#define debugOutput(expr) do {                     \
    std::cout << #expr " = " << expr << std::endl; \
} while(0);

PBRT_NAMESPACE_END

#endif // !_PBRT_DEBUG_H_
