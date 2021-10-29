#ifndef _VEC3_HPP_
#define _VEC3_HPP_

#include <iostream>

#include "utility.hpp"

class vec3 {
public:
    vec3() : v{0.0f, 0.0f, 0.0f} {}
    vec3(float v1, float v2, float v3) : v{v1, v2, v3} {}

    inline static vec3 random_in_unit_cube(float a = -1.0f, float b = 1.0f) {
        return vec3(uniform_float(a, b), uniform_float(a, b), uniform_float(a, b));
    }
    static vec3 random_in_unit_sphere() {
        while (true) {
            vec3 s = vec3::random_in_unit_cube();
            if (s.length_squared() >= 1.0f)
                continue;
            return s;
        }
    }
    static vec3 random_in_unit_circle() {
        while (true) {
            vec3 s = vec3(uniform_float(-1.0f, 1.0f), uniform_float(-1.0f, 1.0f), 0.0f);
            if (s.length_squared() >= 1.0f)
                continue;
            return s;
        }
    }

    float x() const { return v[0]; }
    float y() const { return v[1]; }
    float z() const { return v[2]; }

    vec3 operator+(const vec3& other) const {
        return vec3(v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2]);
    }
    vec3 operator-(const vec3& other) const {
        return vec3(v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2]);
    }
    vec3 operator*(const vec3& other) const {
        return vec3(v[0] * other.v[0], v[1] * other.v[1], v[2] * other.v[2]);
    }
    vec3 operator*(float t) const {
        return vec3(v[0] * t, v[1] * t, v[2] * t);
    }
    vec3 operator/(float t) const {
        return operator*(1.0 / t);
    }
    friend vec3 operator*(float t, const vec3& vec) {
        return vec * t;
    }
    friend vec3 operator/(float t, const vec3& vec) {
        return vec / t;
    }

    vec3 operator-() const { return vec3(-v[1], -v[2], -v[3]); }
    float operator[](int i) const { return v[i]; }
    float& operator[](int i) { return v[i]; }

    vec3& operator+=(const vec3& other) {
        v[0] += other.v[0];
        v[1] += other.v[1];
        v[2] += other.v[2];
        return *this;
    }
    vec3& operator*=(const float t) {
        v[0] *= t;
        v[1] *= t;
        v[2] *= t;
        return *this;
    }
    vec3& operator/=(const float t) {
        return operator*=(1.0f / t);
    }
    vec3& apply_function(std::function<float(float)>& f) {
        v[0] = f(v[0]);
        v[1] = f(v[1]);
        v[2] = f(v[2]);
        return *this;
    }

    float length_squared() const {
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    }
    float length() const {
        return sqrt(length_squared());
    }

    vec3 unit_vector() const {
        return operator/(length());
    }
    float dot(const vec3& other) const {
        return v[0] * other.v[0] + v[1] * other.v[1] + v[2] * other.v[2];
    }
    vec3 cross(const vec3& other) const {
        return vec3(v[1] * other.v[2] - v[2] * other.v[1],
                    v[2] * other.v[0] - v[0] * other.v[2],
                    v[0] * other.v[1] - v[1] * other.v[0]);
    }

    bool near_zero() const {
        const float epsilon = 1e-6f;
        return (fabs(v[0]) < epsilon) && (fabs(v[1]) < epsilon) && (fabs(v[2]) < epsilon);
    }

public:
    float v[3];
};

using point3 = vec3;
using color = vec3;

#endif // !_VEC3_HPP_
