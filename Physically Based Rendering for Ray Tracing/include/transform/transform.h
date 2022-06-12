#ifndef _PBRT_TRANSFORM_H_
#define _PBRT_TRANSFORM_H_

#include "math/square_matrix.h"
#include "../common.h"

PBRT_NAMESPACE_START

class Transform {
public:
    Transform() = default;
    Transform(Transform const& rhs);
    Transform(Float const mat[4][4]);

private:
    SquareMatrix<4> mat;
    SquareMatrix<4> inv_mat;
};

PBRT_NAMESPACE_END

#endif // !_PBRT_TRANSFORM_
