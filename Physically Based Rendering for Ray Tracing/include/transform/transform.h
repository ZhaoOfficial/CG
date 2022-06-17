#ifndef _PBRT_TRANSFORM_H_
#define _PBRT_TRANSFORM_H_

#include "geometry/vector3.h"
#include "math/arithmetic.h"
#include "math/square_matrix.h"
#include "../common.h"

PBRT_NAMESPACE_START

class Transform {
public:
    //! Constructor and destructor
    Transform() = default;
    Transform(SquareMatrix<4> const& mat, SquareMatrix<4> const& inv_mat);
    Transform(Float const arr[16]);
    Transform(Float const arr[4][4]);
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

//! Non-member function
// @param[in] `T`: the translation vector.
Transform translate(Vector3f const& T);
// @param[in] `S`: the scaling factor along each axis.
Transform scale(Vector3f const& S);
// @param[in] `angle`: in radius form.
Transform rotateX(Float angle);
// @param[in] `angle`: in radius form.
Transform rotateY(Float angle);
// @param[in] `angle`: in radius form.
Transform rotateZ(Float angle);
// The view matrix, A.K.A the world to camera matrix.
// @param[in] `pos`: the position the camera currently sits.
// @param[in] `look`: the point that the camera currently looks at.
// @param[in] `ref_up`: a reference up direction.
// @return matrix: the view matrix.
Transform lookAt(Point3f const& pos, Point3f const& look, Vector3f const& ref_up);

PBRT_NAMESPACE_END

#endif // !_PBRT_TRANSFORM_
