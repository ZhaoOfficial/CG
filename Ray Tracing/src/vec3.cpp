#include <iostream>

#include "vec3.hpp"
#include "utility.hpp"

namespace RayTracing {

    vec3::vec3() : v{ 0.0, 0.0, 0.0 } {}
    vec3::vec3(float v1, float v2, float v3) : v{ v1, v2, v3 } {}

    float vec3::x() const { return v[0]; }
    float vec3::y() const { return v[1]; }
    float vec3::z() const { return v[2]; }
    float vec3::operator[](int i) const { return v[i]; }
    float& vec3::x() { return v[0]; }
    float& vec3::y() { return v[1]; }
    float& vec3::z() { return v[2]; }
    float& vec3::operator[](int i) { return v[i]; }

    vec3 vec3::operator+(const vec3& other) const {
        return vec3{ v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2] };
    }
    vec3 vec3::operator-(const vec3& other) const {
        return vec3{ v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2] };
    }
    vec3 vec3::operator-() const { return vec3{ -v[0], -v[1], -v[2] }; }
    vec3 vec3::operator*(const vec3& other) const {
        return vec3{ v[0] * other.v[0], v[1] * other.v[1], v[2] * other.v[2] };
    }
    vec3 vec3::operator*(float t) const {
        return vec3{ v[0] * t, v[1] * t, v[2] * t };
    }
    vec3 vec3::operator/(float t) const {
        return this->operator*(1.0f / t);
    }
    vec3 operator*(float t, const vec3& vec) {
        return vec * t;
    }
    vec3 operator/(float t, const vec3& vec) {
        return vec / t;
    }

    vec3& vec3::operator+=(const vec3& other) {
        v[0] += other.v[0];
        v[1] += other.v[1];
        v[2] += other.v[2];
        return *this;
    }
    vec3& vec3::operator-=(const vec3& other) {
        v[0] -= other.v[0];
        v[1] -= other.v[1];
        v[2] -= other.v[2];
        return *this;
    }
    vec3& vec3::operator*=(const float t) {
        v[0] *= t;
        v[1] *= t;
        v[2] *= t;
        return *this;
    }
    vec3& vec3::operator/=(const float t) {
        return this->operator*=(1.0f / t);
    }

    float vec3::length_squared() const {
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    }
    float vec3::length() const {
        return std::sqrt(length_squared());
    }

    bool vec3::near_zero() const {
        static float epsilon = 1e-6f;
        return (fabs(v[0]) < epsilon) && (fabs(v[1]) < epsilon) && (fabs(v[2]) < epsilon);
    }

    std::ostream& operator<<(std::ostream& out, const vec3& a) {
        return out << a.v[0] << ' ' << a.v[1] << ' ' << a.v[2];
    }

    vec3 sqrt(const vec3& a) {
        return vec3{
            std::sqrt(a.v[0]), std::sqrt(a.v[1]), std::sqrt(a.v[2])
        };
    }
    vec3 unit_vector(const vec3& a) {
        return a / a.length();
    }
    float dot(const vec3& a, const vec3& b) {
        return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2];
    }
    vec3 cross(const vec3& a, const vec3& b) {
        return vec3{
            a.v[1] * b.v[2] - a.v[2] * b.v[1],
            a.v[2] * b.v[0] - a.v[0] * b.v[2],
            a.v[0] * b.v[1] - a.v[1] * b.v[0]
        };
    }

    vec3 random_in_unit_cube(float a, float b) {
        return vec3(uniform_float(a, b), uniform_float(a, b), uniform_float(a, b));
    }
    vec3 random_in_unit_sphere() {
        while (true) {
            vec3 s = vec3{ uniform_float(-1.0f, 1.0f), uniform_float(-1.0f, 1.0f), uniform_float(-1.0f, 1.0f) };
            if (s.length_squared() >= 1.0f)
                continue;
            return s;
        }
    }
    vec3 random_in_unit_circle() {
        while (true) {
            vec3 s = vec3{ uniform_float(-1.0f, 1.0f), uniform_float(-1.0f, 1.0f), 0.0f };
            if (s.length_squared() >= 1.0f)
                continue;
            return s;
        }
    }

}