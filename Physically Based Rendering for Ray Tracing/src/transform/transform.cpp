#include "common.h"
#include "math/square_matrix.h"
#include "transform/transform.h"

PBRT_NAMESPACE_START

// Constructor and destructor
Transform::Transform(SquareMatrix<4> const& mat, SquareMatrix<4> const& inv_mat) : mat{mat}, inv_mat{inv_mat} {}
Transform::Transform(Float const arr[16]) : mat{arr}, inv_mat{mat.inverse()} {}
Transform::Transform(Float const arr[4][4]) : mat{arr}, inv_mat{mat.inverse()} {}

// Operator overloading
bool Transform::operator==(Transform const& rhs) const { return (this->mat == rhs.mat) and (this->inv_mat == rhs.inv_mat); }
bool Transform::operator!=(Transform const& rhs) const { return (this->mat != rhs.mat) or (this->inv_mat != rhs.inv_mat); }
bool Transform::operator<(Transform const& rhs) const { return this->mat < rhs.mat; }

// Auxiliary functions
SquareMatrix<4> const& Transform::getMatrix() const { return this->mat; };
SquareMatrix<4> const& Transform::getInverseMatrix() const { return this->inv_mat; };

// Non-member function
Transform translate(Vector3f const& T) {
    Float mat[16] = {
        1, 0, 0, T.x,
        0, 1, 0, T.y,
        0, 0, 1, T.z,
        0, 0, 0, 1
    };
    Float inv_mat[16] = {
        1, 0, 0, -T.x,
        0, 1, 0, -T.y,
        0, 0, 1, -T.z,
        0, 0, 0, 1
    };
    return Transform{ SquareMatrix<4> {mat}, SquareMatrix<4> {inv_mat} };
}

Transform scale(Vector3f const& S) {
    Float mat[16] = {
        S.x, 0,   0,   0,
        0,   S.y, 0,   0,
        0,   0,   S.z, 0,
        0,   0,   0,   1
    };
    Float inv_mat[16] = {
        1 / S.x, 0,       0,       0,
        0,       1 / S.y, 0,       0,
        0,       0,       1 / S.z, 0,
        0,       0,       0,       1
    };
    return Transform{ SquareMatrix<4> {mat}, SquareMatrix<4> {inv_mat} };
}

Transform rotateX(Float angle) {
    Float sin_angle = std::sin(angle);
    Float cos_angle = std::cos(angle);
    Float arr[16] = {
        1, 0,         0,          0,
        0, cos_angle, -sin_angle, 0,
        0, sin_angle, cos_angle,  0,
        0, 0,         0,          1
    };
    SquareMatrix<4> mat{arr};
    return Transform{ mat, mat.transpose() };
}

Transform rotateY(Float angle) {
    Float sin_angle = std::sin(angle);
    Float cos_angle = std::cos(angle);
    Float arr[16] = {
        cos_angle,  0, sin_angle,  0,
        0,          1, 0,          0,
        -sin_angle, 0, cos_angle, 0,
        0,          0, 0,          1
    };
    SquareMatrix<4> mat{arr};
    return Transform{ mat, mat.transpose() };
}

Transform rotateZ(Float angle) {
    Float sin_angle = std::sin(angle);
    Float cos_angle = std::cos(angle);
    Float arr[16] = {
        cos_angle, -sin_angle, 0, 0, 
        sin_angle, cos_angle,  0, 0, 
        0,         0,          1, 0, 
        0,         0,          0, 1
    };
    SquareMatrix<4> mat{arr};
    return Transform{ mat, mat.transpose() };
}

Transform lookAt(Point3f const& pos, Point3f const& look, Vector3f const& ref_up) {
    Vector3f front = normalized(look - pos);
    Vector3f right = normalized(cross(ref_up, front));
    Vector3f up    = cross(front, right);

    Float arr[16] = {
        right.x, up.x, front.x, pos.x,
        right.y, up.y, front.y, pos.y,
        right.z, up.z, front.z, pos.z,
        0, 0, 0, 1
    };
    SquareMatrix<4> mat{arr};
    return Transform{ mat, mat.inverse() };
}

PBRT_NAMESPACE_END
