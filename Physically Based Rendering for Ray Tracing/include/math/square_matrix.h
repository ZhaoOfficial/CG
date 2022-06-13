#ifndef _PBRT_SQUARE_MATRIX_H_
#define _PBRT_SQUARE_MATRIX_H_

#include <algorithm>

#include "../common.h"

PBRT_NAMESPACE_START

template <std::size_t N>
class SquareMatrix {
public:
    //! Constructor and destructor
    SquareMatrix() = default;
    SquareMatrix(Float const mat[N * N]) {
        for (std::size_t i{}; i < N * N; ++i) {
            this->data_1d[i] = mat[i];
        }
    }
    SquareMatrix(Float const mat[N][N]) {
        for (std::size_t i{}; i < N; ++i) {
            for (std::size_t j{}; j < N; ++j) {
                this->data[i][j] = mat[i][j];
            }
        }
    }
    //! Constructor and destructor

    //! Operator overloading
    //* Arithmetic operators
    SquareMatrix& operator+=(SquareMatrix const& rhs) {
        std::transform(rhs.begin(), rhs.end(), this->begin(), std::plus<Float>{});
        return *this;
    }
    SquareMatrix& operator-=(SquareMatrix const& rhs) {
        std::transform(rhs.begin(), rhs.end(), this->begin(), std::minus<Float>{});
        return *this;
    }
    SquareMatrix& operator*=(Float num) {
        for (auto&& d : this->data_1d) {
            d *= num;
        }
        return *this;
    }
    SquareMatrix& operator/=(Float num) {
        Float inv_num = Float(1) / num;
        return this->operator*=(inv_num);
    }
    friend SquareMatrix operator+(SquareMatrix const& lhs, SquareMatrix const& rhs) {
        auto temp = lhs;
        temp += rhs;
        return temp;
    }
    friend SquareMatrix operator-(SquareMatrix const& lhs, SquareMatrix const& rhs) {
        auto temp = lhs;
        temp -= rhs;
        return temp;
    }
    friend SquareMatrix operator*(SquareMatrix const& lhs, Float rhs) {
        auto temp = lhs;
        temp *= rhs;
        return temp;
    }
    friend SquareMatrix operator*(Float lhs, SquareMatrix const& rhs) {
        return rhs * lhs;
    }
    friend SquareMatrix operator/(SquareMatrix const& lhs, Float rhs) {
        auto temp = lhs;
        temp /= rhs;
        return temp;
    }
    //* Comparation operators
    friend bool operator==(SquareMatrix const& lhs, SquareMatrix const& rhs) {
        return std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }
    friend bool operator!=(SquareMatrix const& lhs, SquareMatrix const& rhs) {
        return !(lhs == rhs);
    }
    friend bool operator<(SquareMatrix const& lhs, SquareMatrix const& rhs) {
        return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }
    //* Indexing operators
    Float& operator[](std::size_t i, std::size_t j) { return this->data[i][j]; }
    Float const& operator[](std::size_t i, std::size_t j) const { return this->data[i][j]; }
    //! Operator overloading

    //! Auxiliary functions
    static SquareMatrix Zeros() {
        return SquareMatrix{};
    }
    static SquareMatrix Ones() {
        SquareMatrix temp;
        std::fill(temp.begin(), temp.end(), Float(1));
        return temp;
    }
    static SquareMatrix Identity() {
        SquareMatrix temp;
        for (std::size_t i{}; i < N * N; i += (N + 1)) {
            temp.data_1d[i] = Float(1);
        }
        return temp;
    }

    bool isIdentity() const {
        // (N - 1) of (1, 0, 0, ...) pattern with (N) (0)s.
        for (std::size_t i = 0; i < N - 1; ++i) {
            if (data_1d[i * N + i] != Float(1)) {
                return false;
            }
            for (std::size_t j = 1; j <= N; ++j) {
                if (data_1d[i * N + i + j] != Float(0)) {
                    return false;
                }
            }
        }
        // last 1.
        if (data_1d[N * N - 1] != Float(1)) {
            return false;
        }
        return true;
    }

    SquareMatrix transpose() const {
        SquareMatrix T;
        for (std::size_t i{}; i < N; ++i) {
            for (std::size_t j{}; j < N; ++j) {
                T[i, j] = this->operator[](j, i);
            }
        }
    }

    Float determinant() const {
        return 0;
    }

    SquareMatrix inverse() const {
        return {};
    }
    //! Auxiliary functions

private:

    Float const* begin() const { return std::begin(this->data_1d); }
    Float const* end() const { return std::end(this->data_1d); }

    union {
        Float data[N][N];
        Float data_1d[N * N] {};
    };
};

//! Specializations of N = 1, 2, 3, 4 are in corresponding `.cpp` file

PBRT_NAMESPACE_END

#endif //!_PBRT_SQUARE_MATRIX_H_
