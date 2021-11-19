#ifndef _VEC3_HPP_
#define _VEC3_HPP_

#include <iostream>
#include "utility.hpp"

namespace RayTracing {

    class Vec3 {
    public:
        // constructor
        Vec3();
        Vec3(float v1, float v2, float v3);

        // access function
        float x() const;
        float y() const;
        float z() const;
        float operator[](int i) const;
        float& x();
        float& y();
        float& z();
        float& operator[](int i);

        // operator overload
        Vec3 operator+(const Vec3& other) const;
        Vec3 operator-(const Vec3& other) const;
        Vec3 operator-() const;
        Vec3 operator*(const Vec3& other) const;
        Vec3 operator*(float t) const;
        Vec3 operator/(float t) const;
        friend Vec3 operator*(float t, const Vec3& vec);
        friend Vec3 operator/(float t, const Vec3& vec);

        Vec3& operator+=(const Vec3& other);
        Vec3& operator-=(const Vec3& other);
        Vec3& operator*=(const float t);
        Vec3& operator/=(const float t);

        // member function
        float length_squared() const;
        float length() const;

        bool near_zero() const;

    public:
        float v[3];
    };

    using Point3 = Vec3;
    using Color = Vec3;

    // overload
    std::ostream& operator<<(std::ostream& out, const Vec3& a);

    // static function
    Vec3 sqrt(const Vec3& a);
    Vec3 unit_vector(const Vec3& a);
    float dot(const Vec3& a, const Vec3& b);
    Vec3 cross(const Vec3& a, const Vec3& b);

    Vec3 random_in_unit_cube(float a = 0.0f, float b = 1.0f);
    Vec3 random_in_unit_sphere();
    Vec3 random_in_unit_circle();

}

#endif // !_VEC3_HPP_
