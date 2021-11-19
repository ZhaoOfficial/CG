#include <iostream>

#include "vec3.hpp"
#include "utility.hpp"

namespace RayTracing {

    Vec3::Vec3() : v{ 0.0, 0.0, 0.0 } {}
    Vec3::Vec3(float v1, float v2, float v3) : v{ v1, v2, v3 } {}

    float Vec3::x() const { return v[0]; }
    float Vec3::y() const { return v[1]; }
    float Vec3::z() const { return v[2]; }
    float Vec3::operator[](int i) const { return v[i]; }
    float& Vec3::x() { return v[0]; }
    float& Vec3::y() { return v[1]; }
    float& Vec3::z() { return v[2]; }
    float& Vec3::operator[](int i) { return v[i]; }

    Vec3 Vec3::operator+(const Vec3& other) const {
        return Vec3{ v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2] };
    }
    Vec3 Vec3::operator-(const Vec3& other) const {
        return Vec3{ v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2] };
    }
    Vec3 Vec3::operator-() const { return Vec3{ -v[0], -v[1], -v[2] }; }
    Vec3 Vec3::operator*(const Vec3& other) const {
        return Vec3{ v[0] * other.v[0], v[1] * other.v[1], v[2] * other.v[2] };
    }
    Vec3 Vec3::operator*(float t) const {
        return Vec3{ v[0] * t, v[1] * t, v[2] * t };
    }
    Vec3 Vec3::operator/(float t) const {
        return this->operator*(1.0f / t);
    }
    Vec3 operator*(float t, const Vec3& vec) {
        return vec * t;
    }
    Vec3 operator/(float t, const Vec3& vec) {
        return vec / t;
    }

    Vec3& Vec3::operator+=(const Vec3& other) {
        v[0] += other.v[0];
        v[1] += other.v[1];
        v[2] += other.v[2];
        return *this;
    }
    Vec3& Vec3::operator-=(const Vec3& other) {
        v[0] -= other.v[0];
        v[1] -= other.v[1];
        v[2] -= other.v[2];
        return *this;
    }
    Vec3& Vec3::operator*=(const float t) {
        v[0] *= t;
        v[1] *= t;
        v[2] *= t;
        return *this;
    }
    Vec3& Vec3::operator/=(const float t) {
        return this->operator*=(1.0f / t);
    }

    float Vec3::length_squared() const {
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    }
    float Vec3::length() const {
        return std::sqrt(length_squared());
    }

    bool Vec3::near_zero() const {
        static float epsilon = 1e-6f;
        return (fabs(v[0]) < epsilon) && (fabs(v[1]) < epsilon) && (fabs(v[2]) < epsilon);
    }

    std::ostream& operator<<(std::ostream& out, const Vec3& a) {
        return out << a.v[0] << ' ' << a.v[1] << ' ' << a.v[2];
    }

    Vec3 sqrt(const Vec3& a) {
        return Vec3{
            std::sqrt(a.v[0]), std::sqrt(a.v[1]), std::sqrt(a.v[2])
        };
    }
    Vec3 unit_vector(const Vec3& a) {
        return a / a.length();
    }
    float dot(const Vec3& a, const Vec3& b) {
        return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2];
    }
    Vec3 cross(const Vec3& a, const Vec3& b) {
        return Vec3{
            a.v[1] * b.v[2] - a.v[2] * b.v[1],
            a.v[2] * b.v[0] - a.v[0] * b.v[2],
            a.v[0] * b.v[1] - a.v[1] * b.v[0]
        };
    }

    Vec3 random_in_unit_cube(float a, float b) {
        return Vec3(uniform_float(a, b), uniform_float(a, b), uniform_float(a, b));
    }
    Vec3 random_in_unit_sphere() {
        while (true) {
            Vec3 s = Vec3{ uniform_float(-1.0f, 1.0f), uniform_float(-1.0f, 1.0f), uniform_float(-1.0f, 1.0f) };
            if (s.length_squared() >= 1.0f)
                continue;
            return s;
        }
    }
    Vec3 random_in_unit_circle() {
        while (true) {
            Vec3 s = Vec3{ uniform_float(-1.0f, 1.0f), uniform_float(-1.0f, 1.0f), 0.0f };
            if (s.length_squared() >= 1.0f)
                continue;
            return s;
        }
    }

}