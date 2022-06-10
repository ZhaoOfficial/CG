#ifndef _PBRT_VECTOR2_H_
#define _PBRT_VECTOR2_H_

#include <iostream>

#include "util/arithmetic.h"
#include "../common.h"

PBRT_NAMESPACE_START

template <typename T>
struct Vector2 {
    // Interface of vector.
    using value_type      = std::decay_t<T>;
    using reference       = value_type&;
    using pointer         = value_type*;
    using const_reference = value_type const&;

    //! Constructor and destructor
    Vector2(T const& x = T{}, T const& y = T{}) noexcept : x(x), y(y) {}
    Vector2(Vector2 const& rhs) = default;
    Vector2& operator=(Vector2 const& rhs) = default;
    Vector2(Vector2&& rhs) = default;
    Vector2& operator=(Vector2&& rhs) = default;
    ~Vector2() = default;
    //! Constructor and destructor

    //! Operator overloading
    //* Arithmetic operators
    Vector2& operator+=(Vector2 const& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        return *this;
    }
    Vector2& operator-=(Vector2 const& rhs) {
        this->x -= rhs.x;
        this->y -= rhs.y;
        return *this;
    }
    template <NumericType Number>
    Vector2& operator*=(Number num) {
        this->x *= num;
        this->y *= num;
        return *this;
    }
    template <NumericType Number>
    Vector2& operator/=(Number num) {
        Number inv_num = Number(1) / num;
        return this->operator*=(inv_num);
    }
    //* Arithmetic operators
    friend Vector2 operator+(Vector2 const& lhs, Vector2 const& rhs) {
        Vector2 temp = lhs;
        temp += rhs;
        return temp;
    }
    friend Vector2 operator-(Vector2 const& lhs, Vector2 const& rhs) {
        Vector2 temp = lhs;
        temp -= rhs;
        return temp;
    }
    friend Vector2 operator-(Vector2 const& rhs) {
        return Vector2(-rhs.x, -rhs.y);
    }
    template <NumericType Number>
    friend Vector2 operator*(Vector2 const& lhs, Number rhs) {
        Vector2 temp = lhs;
        temp *= rhs;
        return temp;
    }
    template <NumericType Number>
    friend Vector2 operator*(Number lhs, Vector2 const& rhs) {
        return rhs * lhs;
    }
    friend Vector2 operator/(Vector2 const& lhs, T rhs) {
        Vector2 temp = lhs;
        temp /= rhs;
        return temp;
    }
    //* Comparation operators
    friend bool operator==(Vector2 const& lhs, Vector2 const& rhs) {
        return (lhs.x == rhs.x) and (lhs.y == rhs.y);
    }
    friend bool operator!=(Vector2 const& lhs, Vector2 const& rhs) {
        return (lhs.x != rhs.x) or (lhs.y != lhs.y);
    }
    //* Indexing operators
    const_reference operator[](std::size_t idx) const {
        return this->data[idx];
    }
    reference operator[](std::size_t idx) {
        return this->data[idx];
    }
    //* Output operator
    friend std::ostream& operator<<(std::ostream& out, Vector2 const& rhs) {
        out << "[ " << rhs.x << ", " << rhs.y << " ]";
        return out;
    }
    //! Operator overloading

    //! Auxiliary functions
    static Vector2 Zeros() {
        return Vector2{};
    }
    static Vector2 Ones() {
        return Vector2(T(1), T(1));
    }
    Float squareNorm() const {
        return this->x * this->x + this->y * this->y;
    }
    Float norm() const {
        return std::sqrt(squareNorm());
    }
    value_type min() const {
        return std::min(this->x, this->y);
    }
    value_type max() const {
        return std::max(this->x, this->y);
    }
    friend Vector2 min(Vector2 const& lhs, Vector2 const& rhs) {
        return Vector2 {
            std::min(lhs.x, rhs.x),
            std::min(lhs.y, rhs.y)
        };
    }
    friend Vector2 max(Vector2 const& lhs, Vector2 const& rhs) {
        return Vector2 {
            std::max(lhs.x, rhs.x),
            std::max(lhs.y, rhs.y)
        };
    }
    friend Float dot(Vector2 const& lhs, Vector2 const& rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y;
    }
    friend Vector2 abs(Vector2 const& rhs) {
        return Vector2{ std::abs(rhs.x), std::abs(rhs.y) };
    }
    friend Vector2 normalized(Vector2 const& rhs) {
        return rhs / rhs.norm();
    }
    //! Auxiliary functions

    // Member variables
    union {
        T data[2];
        struct { T x, y; };
    };
};

using Vector2i = Vector2<int>;
using Vector2f = Vector2<Float>;

PBRT_NAMESPACE_END

#endif // !_PBRT_VECTOR2_H_
