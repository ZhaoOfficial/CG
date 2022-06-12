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
    SquareMatrix& operator+=(SquareMatrix const& rhs) {
        std::transform(rhs.begin(), rhs.end(), this->begin(), std::plus<Float>{});
        return *this;
    }
    SquareMatrix& operator-=(SquareMatrix const& rhs) {
        std::transform(rhs.begin(), rhs.end(), this->begin(), std::minus<Float>{});
        return *this;
    }
    SquareMatrix& operator*=(Float num) {
        for (auto&& d : *this) {
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
    friend bool operator==(SquareMatrix const& lhs, SquareMatrix const& rhs) {
        return std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }
    friend bool operator!=(SquareMatrix const& lhs, SquareMatrix const& rhs) {
        return !(lhs == rhs);
    }
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

//! Specialization of N = 1, 2, 3, 4
//* N = 1
template <>
Float SquareMatrix<1>::determinant() const {
    return this->data[0][0];
}
template <>
SquareMatrix<1> SquareMatrix<1>::inverse() const {
    if (this->data[0][0] == Float(0)) {
        return SquareMatrix<1>{};
    }
    Float mat[1] = { Float(1) / this->data[0][0] };
    return SquareMatrix<1>{mat};
}

//* N = 2
template <>
Float SquareMatrix<2>::determinant() const {
    Float a1 = this->data[0][0] * this->data[1][1];
    Float b1 = this->data[0][1] * this->data[1][0];

    return a1 - b1;
}
template <>
SquareMatrix<2> SquareMatrix<2>::inverse() const {
    Float det = this->determinant();
    if (det == 0) {
        return SquareMatrix<2>{};
    }
    Float inv_det = Float(1) / det;

    Float mat[4] {
         this->data[1][1], -this->data[0][1],
        -this->data[1][0],  this->data[0][0]
    };
    for (auto&& i : mat) {
        i *= inv_det;
    }
    
    return SquareMatrix<2>{mat};
}

//* N = 3
template <>
Float SquareMatrix<3>::determinant() const {
    Float a1 = this->data[0][0] * this->data[1][1] * this->data[2][2];
    Float a2 = this->data[0][1] * this->data[1][2] * this->data[2][0];
    Float a3 = this->data[0][2] * this->data[1][0] * this->data[2][1];

    Float b1 = this->data[0][0] * this->data[1][2] * this->data[2][1];
    Float b2 = this->data[0][1] * this->data[1][0] * this->data[2][2];
    Float b3 = this->data[0][2] * this->data[1][1] * this->data[2][0];

    return a1 + a2 + a3 - b1 - b2 - b3;
}
template <>
SquareMatrix<3> SquareMatrix<3>::inverse() const {
    Float det = this->determinant();
    if (det == 0) {
        return SquareMatrix<3>{};
    }
    Float inv_det = Float(1) / det;

    Float mat[9] {
         this->data[1][1], -this->data[0][1],
        -this->data[1][0],  this->data[0][0]
    };
    for (auto&& i : mat) {
        i *= inv_det;
    }
    
    return SquareMatrix<3>{mat} * inv_det;
}

//! Specialization of N = 1, 2, 3, 4

PBRT_NAMESPACE_END

#endif //!_PBRT_SQUARE_MATRIX_H_
