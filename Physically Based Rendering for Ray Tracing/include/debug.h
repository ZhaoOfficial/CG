#ifndef _PBRT_DEBUG_H_
#define _PBRT_DEBUG_H_

#define debugOutput(expr) do {                     \
    std::cout << #expr " = " << expr << std::endl; \
} while(0);

#endif // !_PBRT_DEBUG_H_
