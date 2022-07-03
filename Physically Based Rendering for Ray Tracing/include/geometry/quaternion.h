#ifndef _PBRT_QUATERNION_H_
#define _PBRT_QUATERNION_H_

#include "vector3.h"
#include "../common.h"

PBRT_NAMESPACE_START

// A class that represents the rotation as a quaternion.
struct Quaternion {
    //! Constructor and destructor
    Quaternion() = default;
    explicit Quaternion(Vector3f const& w, Float u);
    //! Constructor and destructor

    //! Operator overloading
    Quaternion& operator+=(Quaternion const& rhs);
    Quaternion& operator-=(Quaternion const& rhs);
    Quaternion& operator*=(Float f);
    Quaternion& operator/=(Float f);

    friend Quaternion operator+(Quaternion const& lhs, Quaternion const& rhs);
    friend Quaternion operator-(Quaternion const& lhs, Quaternion const& rhs);
    friend Quaternion operator-(Quaternion const& rhs);
    friend Quaternion operator*(Quaternion const& lhs, Float f);
    friend Quaternion operator*(Float f, Quaternion const& rhs);
    friend Quaternion operator/(Quaternion const& lhs, Float f);

    friend bool operator==(Quaternion const& lhs, Quaternion const& rhs);
    friend bool operator!=(Quaternion const& lhs, Quaternion const& rhs);
    friend std::ostream& operator<<(std::ostream& out, Quaternion const& rhs);
    //! Operator overloading

    //! Auxiliary functions
    // The square of the L2 norm.
    Float squareNorm() const;
    // The L2 norm.
    Float norm() const;
    //! Auxiliary functions

    Vector3f w{};
    Float u{1};
};

// Return the dot-product of two quaternion.
Float dot(Quaternion const& lhs, Quaternion const& rhs);
// Out-place version of normalization.
Quaternion normalized(Quaternion const& rhs);
// Angle between two quaternions.
Float angleBetween(Quaternion const& lhs, Quaternion const& rhs);
// Interpolate between two quaternions, lerping on the surface of the sphere.
Quaternion slerp(Float t, Quaternion const& lhs, Quaternion const& rhs);

PBRT_NAMESPACE_END

#endif // !_PBRT_QUATERNION_H_
