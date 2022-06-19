#ifndef _PBRT_TRANSFORM_H_
#define _PBRT_TRANSFORM_H_

#include <array>
#include <cassert>

#include "geometry/ray.h"
#include "geometry/vector3.h"
#include "math/arithmetic.h"
#include "math/square_matrix.h"
#include "../common.h"

PBRT_NAMESPACE_START

// A class that stores the transformation matrices.
// The default transform is identity.
class Transform {
public:
    //! Constructor and destructor
    Transform() = default;
    Transform(SquareMatrix<4> const& mat, SquareMatrix<4> const& inv_mat);
    Transform(std::array<Float, 16> const& arr);
    Transform(std::array<std::array<Float, 4>, 4> const& arr);
    //! Constructor and destructor

    //! Operator overloading
    bool operator==(Transform const& rhs) const;
    bool operator!=(Transform const& rhs) const;
    bool operator<(Transform const& rhs) const;

    friend std::ostream& operator<<(std::ostream& out, Transform const& rhs);
    //! Operator overloading

    //! Auxiliary functions
    // Return a new `Transform` that is the inverse of this transform.
    Transform inverse() const;
    // Return a new `Transform` that is the transpose of this transform.
    Transform transpose() const;

    SquareMatrix<4> const& getMatrix() const;
    SquareMatrix<4> const& getInverseMatrix() const;

    // Check if the transform is identitical.
    bool isIdentity() const;
    // Check if the determinant of the transform is near to 1.
    bool hasScale(Float tolerance) const;
    // Perform a linear transformation on the given vector.
    Vector3f operator()(Vector3f const& v) const;
    //! Auxiliary functions

private:
    SquareMatrix<4> mat{std::array<Float, 16> {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
    SquareMatrix<4> inv_mat{std::array<Float, 16> {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
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
