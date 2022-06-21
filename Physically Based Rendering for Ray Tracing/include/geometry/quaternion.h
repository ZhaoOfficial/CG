#ifndef _PBRT_QUATERNION_H_
#define _PBRT_QUATERNION_H_

#include "vector3.h"
#include "../common.h"

PBRT_NAMESPACE_START

class Quaternion {
public:
    Quaternion() = default;

private:
    Vector3f w;
    Float u;
};

PBRT_NAMESPACE_END

#endif // !_PBRT_QUATERNION_H_
