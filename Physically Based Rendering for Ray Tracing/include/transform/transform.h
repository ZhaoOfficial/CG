#ifndef _PBRT_TRANSFORM_H_
#define _PBRT_TRANSFORM_H_

#include "math/square_matrix.h"
#include "../common.h"

PBRT_NAMESPACE_START

class Transform {
public:
    //! Constructor and destructor
    Transform() = default;
    Transform(Float const mat[16]);
    Transform(Float const mat[4][4]);
    //! Constructor and destructor

    //! Operator overloading
    bool operator==(Transform const& rhs) const;
    bool operator!=(Transform const& rhs) const;
    bool operator<(Transform const& rhs) const;
    //! Operator overloading

    //! Auxiliary functions
    SquareMatrix<4> const& getMatrix() const;
    SquareMatrix<4> const& getInverseMatrix() const;
    //! Auxiliary functions

private:
    SquareMatrix<4> mat;
    SquareMatrix<4> inv_mat;
};

PBRT_NAMESPACE_END

#endif // !_PBRT_TRANSFORM_
