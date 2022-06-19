#include "transform/transform.h"

PBRT_NAMESPACE_START

// Constructor and destructor
Transform::Transform(SquareMatrix<4> const& mat, SquareMatrix<4> const& inv_mat) : mat{mat}, inv_mat{inv_mat} {}
Transform::Transform(std::array<Float, 16> const& arr) : mat{arr}, inv_mat{mat.inverse()} {}
Transform::Transform(std::array<std::array<Float, 4>, 4> const& arr) : mat{arr}, inv_mat{mat.inverse()} {}

// Operator overloading
bool Transform::operator==(Transform const& rhs) const { return this->mat == rhs.mat; }
bool Transform::operator!=(Transform const& rhs) const { return this->mat != rhs.mat; }
bool Transform::operator<(Transform const& rhs) const { return this->mat < rhs.mat; }

std::ostream& operator<<(std::ostream& out, Transform const& rhs) {
    out << rhs.mat;
    return out;
}
// Auxiliary functions
Transform Transform::inverse() const { return Transform{this->inv_mat, this->mat}; }
Transform Transform::transpose() const { return Transform{this->mat.transpose(), this->inv_mat.transpose()}; }

SquareMatrix<4> const& Transform::getMatrix() const { return this->mat; };
SquareMatrix<4> const& Transform::getInverseMatrix() const { return this->inv_mat; };

bool Transform::isIdentity() const { return this->mat.isIdentity(); }

bool Transform::hasScale(Float tolerance) const {
    Float x = this->operator()(Vector3f{1, 0, 0}).squareNorm();
    Float y = this->operator()(Vector3f{0, 1, 0}).squareNorm();
    Float z = this->operator()(Vector3f{0, 0, 1}).squareNorm();
    return (
        std::abs(x - Float(1.0)) > tolerance ||
        std::abs(y - Float(1.0)) > tolerance ||
        std::abs(z - Float(1.0)) > tolerance
    );
}

Vector3f Transform::operator()(Vector3f const& v) const {
    return Vector3f{
        this->mat[{0, 0}] * v.x + this->mat[{0, 1}] * v.y + this->mat[{0, 2}] * v.z,
        this->mat[{1, 0}] * v.x + this->mat[{1, 1}] * v.y + this->mat[{1, 2}] * v.z,
        this->mat[{2, 0}] * v.x + this->mat[{2, 1}] * v.y + this->mat[{2, 2}] * v.z
    };
}

// Non-member function
Transform translate(Vector3f const& T) {
    return Transform{ std::array<Float, 16> {
        1, 0, 0, T.x,
        0, 1, 0, T.y,
        0, 0, 1, T.z,
        0, 0, 0, 1
    }, std::array<Float, 16> {
        1, 0, 0, -T.x,
        0, 1, 0, -T.y,
        0, 0, 1, -T.z,
        0, 0, 0, 1
    } };
}

Transform scale(Vector3f const& S) {
    Float mat[16] = {
        S.x, 0,   0,   0,
        0,   S.y, 0,   0,
        0,   0,   S.z, 0,
        0,   0,   0,   1
    };
    assert(S.x != Float(0));
    assert(S.y != Float(0));
    assert(S.z != Float(0));
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
