#ifndef _PBRT_TUPLE2_H_
#define _PBRT_TUPLE2_H_

#include <iostream>
#include <type_traits>

#include "util/arithmetic.h"
#include "../common.h"

PBRT_NAMESPACE_START

// Yes, a template template class
template <template <typename> typename Container, typename T>
struct Tuple2 {
    // Interface of all two element containers.
    using value_type      = std::decay_t<T>;
    using reference       = value_type&;
    using pointer         = value_type*;
    using const_reference = value_type const&;
    static constexpr std::size_t n_dims = 2;

    //! Constructor and destructor
    Tuple2() = default;
    explicit Tuple2(T const& x, T const& y) noexcept : x{x}, y{y} {}
    //! Constructor and destructor

    //! Operator overloading
    //* Arithmetic operators
    template <typename U>
    Container<T>& operator+=(Container<U> const& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        return static_cast<Container<T> &>(*this);
    }
    template <typename U>
    Container<T>& operator-=(Container<U> const& rhs) {
        this->x -= rhs.x;
        this->y -= rhs.y;
        return static_cast<Container<T> &>(*this);
    }
    template <NumericType Number>
    Container<decltype(T{} / Number{})>& operator*=(Number num) {
        this->x *= num;
        this->y *= num;
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
        return Container<T>(-rhs.x, -rhs.y);
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
        return (lhs.x == rhs.x) and (lhs.y == rhs.y);
    }
    friend bool operator!=(Container<T> const& lhs, Container<T> const& rhs) {
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
    friend std::ostream& operator<<(std::ostream& out, Tuple2 const& rhs) {
        out << "[" << rhs.x << ", " << rhs.y << "]";
        return out;
    }
    //! Operator overloading

    //! Auxiliary functions
    static Container<T> Zeros() {
        return Container<T>{};
    }
    static Container<T> Ones() {
        return Container<T>{ T(1), T(1) };
    }
    value_type squareNorm() const { return this->x * this->x + this->y * this->y; }
    Float norm() const { return std::sqrt(Float(squareNorm())); }
    value_type min() const { return std::min(this->x, this->y); }
    value_type max() const { return std::max(this->x, this->y); }
    template <typename U>
    Container<std::common_type_t<T, U>> cwiseMin(Container<U> const& rhs) const {
        return Container<std::common_type_t<T, U>> {
            std::min<std::common_type_t<T, U>>(this->x, rhs.x),
            std::min<std::common_type_t<T, U>>(this->y, rhs.y)
        };
    }
    template <typename U>
    Container<std::common_type_t<T, U>> cwiseMax(Container<U> const& rhs) const {
        return Container<std::common_type_t<T, U>> {
            std::max<std::common_type_t<T, U>>(this->x, rhs.x),
            std::max<std::common_type_t<T, U>>(this->y, rhs.y)
        };
    }
    template <typename U>
    Container<decltype(T{} * U{})> cwiseProd(Container<U> const& rhs) const {
        return Container<decltype(T{} * U{})> {
            this->x * rhs.x,
            this->y * rhs.y
        };
    }
    template <typename U>
    Container<decltype(T{} / U{})> cwiseDiv(Container<U> const& rhs) const {
        return Container<decltype(T{} / U{})> {
            this->x / rhs.x,
            this->y / rhs.y
        };
    }
    template <typename U>
    friend decltype(T{} * U{}) dot(Container<T> const& lhs, Container<U> const& rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y;
    }
    friend Container<T> abs(Container<T> const& rhs) {
        return Container<T>{ std::abs(rhs.x), std::abs(rhs.y) };
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
        T data[2] { T{}, T{} };
        struct { T x, y; };
    };
};

PBRT_NAMESPACE_END

#endif // !_PBRT_TUPLE2_H_
