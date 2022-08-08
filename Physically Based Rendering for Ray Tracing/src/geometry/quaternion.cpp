#include <cassert>

#include "geometry/quaternion.h"
#include "math/arithmetic.h"

PBRT_NAMESPACE_START

Quaternion::Quaternion(Vector3f const& w, Float u) : w{w}, u{u} {}

Quaternion& Quaternion::operator+=(Quaternion const& rhs) {
    this->w += rhs.w;
    this->u += rhs.u;
    return *this;
}
Quaternion& Quaternion::operator-=(Quaternion const& rhs) {
    this->w -= rhs.w;
    this->u -= rhs.u;
    return *this;
}
Quaternion& Quaternion::operator*=(Float f) {
    this->w *= f;
    this->u *= f;
    return *this;
}
Quaternion& Quaternion::operator/=(Float f) {
    assert(f != Float{0});
    Float inv_f = Float{1} / f;
    this->w *= inv_f;
    this->u *= inv_f;
    return *this;
}

Quaternion operator+(Quaternion const& lhs, Quaternion const& rhs) {
    return Quaternion(lhs) += rhs;
}
Quaternion operator-(Quaternion const& lhs, Quaternion const& rhs) {
    return Quaternion(lhs) -= rhs;
}
Quaternion operator-(Quaternion const& rhs) {
    return Quaternion(-rhs.w, -rhs.u);
}
Quaternion operator*(Quaternion const& lhs, Float f) {
    return Quaternion(lhs) *= f;
}
Quaternion operator*(Float f, Quaternion const& rhs) {
    return Quaternion(rhs) *= f;
}
Quaternion operator/(Quaternion const& lhs, Float f) {
    return Quaternion(lhs) /= f;
}

bool operator==(Quaternion const& lhs, Quaternion const& rhs) {
    return (lhs.w == rhs.w) && (lhs.u == rhs.u);
}
bool operator!=(Quaternion const& lhs, Quaternion const& rhs) {
    return (lhs.w != rhs.w) || (lhs.u != rhs.u);
}
std::ostream& operator<<(std::ostream& out, Quaternion const& rhs) {
    out << "[" << rhs.w.x << ", " << rhs.w.y << ", " << rhs.w.z << ", " << rhs.u << "]";
    return out;
}

Float Quaternion::squareNorm() const {
    return dot(*this, *this);
}

Float Quaternion::norm() const {
    return std::sqrt(this->squareNorm());
}

Float dot(Quaternion const& lhs, Quaternion const& rhs) {
    return dot(lhs.w, rhs.w) + lhs.u * rhs.u;
}

Quaternion normalized(Quaternion const& rhs) {
    return rhs / rhs.norm();
}

Float angleBetween(Quaternion const& lhs, Quaternion const& rhs) {
    if (dot(lhs, rhs) < Float{0}) {
        return PI - Float{2} * std::asin((lhs + rhs).norm() / Float{2});
    }
    else {
        return Float{2} * std::asin((rhs - lhs).norm() / Float{2});
    }
}

Quaternion slerp(Float t, Quaternion const& lhs, Quaternion const& rhs) {
    Float angle = angleBetween(lhs, rhs);
    Float sinc_angle = sinc(angle);
    return (
        lhs * (Float{1} - t) * sinc((Float{1} - t) * angle) / sinc_angle + rhs * t * sinc(t * angle) / sinc_angle
    );
}

PBRT_NAMESPACE_END
