#ifndef _PBRT_TUPLE3_H_
#define _PBRT_TUPLE3_H_

#include <iostream>
#include <type_traits>

#include "util/arithmetic.h"
#include "../common.h"

PBRT_NAMESPACE_START

// Yes, a template template class
template <template <typename> typename Container, typename T>
struct Tuple3 {
    // Interface of all two element containers.
    using value_type      = std::decay_t<T>;
    using reference       = value_type&;
    using pointer         = value_type*;
    using const_reference = value_type const&;
    static constexpr std::size_t n_dims = 3;

    //! Constructor and destructor
    Tuple3() = default;
    explicit Tuple3(T const& x, T const& y, T const& z) noexcept : x{x}, y{y}, z{z} {}
    //! Constructor and destructor

    //! Operator overloading
    //* Arithmetic operators
    template <typename U>
    Container<T>& operator+=(Container<U> const& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        this->z += rhs.z;
        return static_cast<Container<T> &>(*this);
    }
    template <typename U>
    Container<T>& operator-=(Container<U> const& rhs) {
        this->x -= rhs.x;
        this->y -= rhs.y;
        this->z -= rhs.z;
        return static_cast<Container<T> &>(*this);
    }
    template <NumericType Number>
    Container<decltype(T{} * Number{})>& operator*=(Number num) {
        this->x *= num;
        this->y *= num;
        this->z *= num;
        return static_cast<Container<decltype(T{} * Number{})> &>(*this);
    }
    template <NumericType Number>
    Container<decltype(T{} / Number{})>& operator/=(Number num) {
        auto inv_num = T(1) / num;
        return this->operator*=(inv_num);
    }
    //* Arithmetic operators
    template <typename U>
    friend Container<decltype(T{} + U{})> operator+(Container<T> const& lhs, Container<U> const& rhs)  {
        Container<T> temp = lhs;
        temp += rhs;
        return temp;
    }
    template <typename U>
    friend Container<decltype(T{} + U{})> operator-(Container<T> const& lhs, Container<U> const& rhs) {
        Container<T> temp = lhs;
        temp -= rhs;
        return temp;
    }
    friend Container<T> operator-(Container<T> const& rhs) {
        return Container<T>(-rhs.x, -rhs.y, -rhs.z);
    }
    template <NumericType Number>
    friend Container<decltype(T{} * Number{})> operator*(Container<T> const& lhs, Number rhs) {
        Container<T> temp = lhs;
        temp *= rhs;
        return temp;
    }
    template <NumericType Number>
    friend Container<decltype(T{} * Number{})> operator*(Number lhs, Container<T> const& rhs) {
        return rhs * lhs;
    }
    template <NumericType Number>
    friend Container<decltype(T{} / Number{})> operator/(Container<T> const& lhs, Number rhs) {
        Container<T> temp = lhs;
        temp /= rhs;
        return temp;
    }
    //* Comparation operators
    friend bool operator==(Container<T> const& lhs, Container<T> const& rhs) {
        return (lhs.x == rhs.x) and (lhs.y == rhs.y) and (lhs.z == rhs.z);
    }
    friend bool operator!=(Container<T> const& lhs, Container<T> const& rhs) {
        return (lhs.x != rhs.x) or (lhs.y != lhs.y)or (lhs.z != rhs.z);
    }
    //* Indexing operators
    const_reference operator[](std::size_t idx) const {
        return this->data[idx];
    }
    reference operator[](std::size_t idx) {
        return this->data[idx];
    }
    //* Output operator
    friend std::ostream& operator<<(std::ostream& out, Tuple3 const& rhs) {
        out << "[" << rhs.x << ", " << rhs.y << ", " << rhs.z << "]";
        return out;
    }
    //! Operator overloading

    //! Auxiliary functions
    static Container<T> Zeros() {
        return Container<T>{};
    }
    static Container<T> Ones() {
        return Container<T>{ T(1), T(1), T(1) };
    }
    value_type squareNorm() const { return this->x * this->x + this->y * this->y + this->z * this->z; }
    Float norm() const { return std::sqrt(Float(squareNorm())); }
    value_type min() const { return std::min(std::min(this->x, this->y), this->z); }
    value_type max() const { return std::max(std::max(this->x, this->y), this->z); }
    template <typename U>
    Container<std::common_type_t<T, U>> cwiseMin(Container<U> const& rhs) const {
        return Container<std::common_type_t<T, U>> {
            std::min<std::common_type_t<T, U>>(this->x, rhs.x),
            std::min<std::common_type_t<T, U>>(this->y, rhs.y),
            std::min<std::common_type_t<T, U>>(this->z, rhs.z)
        };
    }
    template <typename U>
    Container<std::common_type_t<T, U>> cwiseMax(Container<U> const& rhs) const {
        return Container<std::common_type_t<T, U>> {
            std::max<std::common_type_t<T, U>>(this->x, rhs.x),
            std::max<std::common_type_t<T, U>>(this->y, rhs.y),
            std::max<std::common_type_t<T, U>>(this->z, rhs.z)
        };
    }
    template <typename U>
    Container<decltype(T{} * U{})> cwiseProd(Container<U> const& rhs) const {
        return Container<decltype(T{} * U{})> {
            this->x * rhs.x,
            this->y * rhs.y,
            this->z * rhs.z
        };
    }
    template <typename U>
    Container<decltype(T{} / U{})> cwiseDiv(Container<U> const& rhs) const {
        return Container<decltype(T{} / U{})> {
            this->x / rhs.x,
            this->y / rhs.y,
            this->z / rhs.z
        };
    }
    template <typename U>
    friend decltype(T{} * U{}) dot(Container<T> const& lhs, Container<U> const& rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }
    template <typename U>
    friend Container<decltype(T{} * U{})> cross(Container<T> const& lhs, Container<U> const& rhs) {
        return Container<decltype(T{} * U{})> {
            lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x
        };
    }
    friend Container<T> abs(Container<T> const& rhs) {
        return Container<T>{ std::abs(rhs.x), std::abs(rhs.y), std::abs(rhs.z) };
    }
    friend Container<T> normalized(Container<T> const& rhs) {
        return rhs / rhs.norm();
    }
    friend Container<T> lerp(Container<T> const& t, Container<T> const& src, Container<T> const& dst) {
        return (dst - src).cwiseProd(t) + src;
    }
    //! Auxiliary functions

    // Member variables
    union {
        T data[3] { T{}, T{}, T{} };
        struct { T x, y, z; };
    };
};

PBRT_NAMESPACE_END

#endif // !_PBRT_TUPLE3_H_
