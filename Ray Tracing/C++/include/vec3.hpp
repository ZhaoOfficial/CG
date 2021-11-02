#ifndef _VEC3_HPP_
#define _VEC3_HPP_

#include <iostream>

#include "utility.hpp"

namespace RayTracing {

    class vec3 {
    public:
        // constructor
        vec3();
        vec3(float v1, float v2, float v3);

        // access function
        inline float x() const;
        inline float y() const;
        inline float z() const;
        float operator[](int i) const;
        float& operator[](int i);

        // operator overload
        vec3 operator+(const vec3& other) const;
        vec3 operator-(const vec3& other) const;
        vec3 operator-() const;
        vec3 operator*(const vec3& other) const;
        vec3 operator*(float t) const;
        vec3 operator/(float t) const;
        friend vec3 operator*(float t, const vec3& vec);
        friend vec3 operator/(float t, const vec3& vec);

        vec3& operator+=(const vec3& other);
        vec3& operator-=(const vec3& other);
        vec3& operator*=(const float t);
        vec3& operator/=(const float t);

        // member function
        float length_squared() const;
        float length() const;

        bool near_zero() const;

    public:
        float v[3];
    };

    using point3 = vec3;
    using color = vec3;

    // overload
    std::ostream& operator<<(std::ostream& out, const vec3& a) {
        return out << a.v[0] << ' ' << a.v[1] << ' ' << a.v[2];
    }

    // static function
    vec3 sqrt(const vec3& a);
    vec3 unit_vector(const vec3& a);
    float dot(const vec3& a, const vec3& b);
    vec3 cross(const vec3& a, const vec3& b);

    inline vec3 random_in_unit_cube(float a = 0.0f, float b = 1.0f);
    vec3 random_in_unit_sphere();
    vec3 random_in_unit_circle();

}

#endif // !_VEC3_HPP_
