#ifndef _PBRT_VECTOR3_H_
#define _PBRT_VECTOR3_H_

#include <iostream>

#include "util/arithmetic.h"
#include "../common.h"

PBRT_NAMESPACE_START

template <typename T>
struct Vector3 {
    // Interface of vector.
    using value_type      = std::decay_t<T>;
    using reference       = value_type&;
    using pointer         = value_type*;
    using const_reference = value_type const&;

    //! Constructor and destructor
    Vector3(T const& x = T{}, T const& y = T{}, T const& z = T{}) noexcept : x(x), y(y), z(z) {}
    Vector3(Vector3 const& rhs) = default;
    Vector3& operator=(Vector3 const& rhs) = default;
    Vector3(Vector3&& rhs) = default;
    Vector3& operator=(Vector3&& rhs) = default;
    ~Vector3() = default;
    //! Constructor and destructor

    //! Operator overloading
    //* Arithmetic operators
    Vector3& operator+=(Vector3 const& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        this->z += rhs.z;
        return *this;
    }
    Vector3& operator-=(Vector3 const& rhs) {
        this->x -= rhs.x;
        this->y -= rhs.y;
        this->z -= rhs.z;
        return *this;
    }
    template <NumericType Number>
    Vector3& operator*=(Number num) {
        this->x *= num;
        this->y *= num;
        this->z *= num;
        return *this;
    }
    template <NumericType Number>
    Vector3& operator/=(Number num) {
        Number inv_num = Number(1) / num;
        return this->operator*=(inv_num);
    }
    //* Arithmetic operators
    friend Vector3 operator+(Vector3 const& lhs, Vector3 const& rhs) {
        Vector3 temp = lhs;
        temp += rhs;
        return temp;
    }
    friend Vector3 operator-(Vector3 const& lhs, Vector3 const& rhs) {
        Vector3 temp = lhs;
        temp -= rhs;
        return temp;
    }
    friend Vector3 operator-(Vector3 const& rhs) {
        return Vector3(-rhs.x, -rhs.y, -rhs.z);
    }
    template <NumericType Number>
    friend Vector3 operator*(Vector3 const& lhs, Number rhs) {
        Vector3 temp = lhs;
        temp *= rhs;
        return temp;
    }
    template <NumericType Number>
    friend Vector3 operator*(Number lhs, Vector3 const& rhs) {
        return rhs * lhs;
    }
    friend Vector3 operator/(Vector3 const& lhs, T rhs) {
        Vector3 temp = lhs;
        temp /= rhs;
        return temp;
    }
    //* Comparation operators
    friend bool operator==(Vector3 const& lhs, Vector3 const& rhs) {
        return (lhs.x == rhs.x) and (lhs.y == rhs.y) and (lhs.z == rhs.z);
    }
    friend bool operator!=(Vector3 const& lhs, Vector3 const& rhs) {
        return (lhs.x != rhs.x) or (lhs.y != lhs.y) or (lhs.z != rhs.z);
    }
    //* Indexing operators
    const_reference operator[](std::size_t idx) const {
        return this->data[idx];
    }
    reference operator[](std::size_t idx) {
        return this->data[idx];
    }
    //* Output operator
    friend std::ostream& operator<<(std::ostream& out, Vector3 const& rhs) {
        out << "[ " << rhs.x << ", " << rhs.y << ", " << rhs.z << " ]";
        return out;
    }
    //! Operator overloading

    //! Auxiliary functions
    static Vector3 Zeros() {
        return Vector3{};
    }
    static Vector3 Ones() {
        return Vector3{ T(1), T(1), T(1) };
    }
    Float squareNorm() const {
        return this->x * this->x + this->y * this->y + this->z * this->z;
    }
    Float norm() const {
        return std::sqrt(squareNorm());
    }
    value_type min() const {
        return std::min(std::min(this->x, this->y), this->z);
    }
    value_type max() const {
        return std::max(std::max(this->x, this->y), this->z);
    }
    friend Vector3 min(Vector3 const& lhs, Vector3 const& rhs) {
        return Vector3 {
            std::min(lhs.x, rhs.x),
            std::min(lhs.y, rhs.y),
            std::min(lhs.z, rhs.z)
        };
    }
    friend Vector3 max(Vector3 const& lhs, Vector3 const& rhs) {
        return Vector3 {
            std::max(lhs.x, rhs.x),
            std::max(lhs.y, rhs.y),
            std::max(lhs.z, rhs.z)
        };
    }
    friend Float dot(Vector3 const& lhs, Vector3 const& rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }
    friend Vector3 abs(Vector3 const& rhs) {
        return Vector3{ std::abs(rhs.x), std::abs(rhs.y), std::abs(rhs.z) };
    }
    friend Vector3 cross(Vector3 const& lhs, Vector3 const& rhs) {
        return Vector3 {
            lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x
        };
    }
    friend Vector3 normalized(Vector3 const& rhs) {
        return rhs / rhs.norm();
    }
    //! Auxiliary functions

    // Member variables
    union {
        T data[3];
        struct { T x, y, z; };
    };
};

using Vector3i = Vector3<int>;
using Vector3f = Vector3<Float>;

PBRT_NAMESPACE_END

#endif // !_PBRT_VECTOR3_H_
