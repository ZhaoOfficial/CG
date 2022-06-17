#ifndef _PBRT_SQUARE_MATRIX_H_
#define _PBRT_SQUARE_MATRIX_H_

#include <algorithm>
#include <functional>

#include "matrix.h"
#include "geometry/vector2.h"
#include "../common.h"

PBRT_NAMESPACE_START

static char const* format[] = {
    "[[%-.4g]]",
    "\n[[%.4g, %.4g]\n [%.4g, %.4g]]",
    "\n[[%.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g]]",
    "\n[[%.4g, %.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g, %.4g]]"
};

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
        std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::plus<Float>{});
        return *this;
    }
    SquareMatrix& operator-=(SquareMatrix const& rhs) {
        std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::minus<Float>{});
        return *this;
    }
    SquareMatrix& operator*=(Float num) {
        std::transform(this->begin(), this->end(), this->begin(), std::bind(std::multiplies{}, std::placeholders::_1, num));
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
    constexpr Float& operator[](Vector2<std::size_t> const& idx) { return this->data[idx.x][idx.y]; }
    constexpr Float const& operator[](Vector2<std::size_t> const& idx) const { return this->data[idx.x][idx.y]; }
    friend std::ostream& operator<<(std::ostream& out, SquareMatrix const& rhs) {
        char buffer[512];
        if constexpr (N == 1) {
            std::sprintf(buffer, format[N - 1], rhs.data[0][0]);
            out << buffer;
        }
        else if constexpr (N == 2) {
            std::sprintf(buffer, format[N - 1], rhs.data[0][0], rhs.data[0][1], rhs.data[1][0], rhs.data[1][1]);
            out << buffer;
        }
        else if constexpr (N == 3) {
            std::sprintf(
                buffer, format[N - 1],
                rhs.data[0][0], rhs.data[0][1], rhs.data[0][2],
                rhs.data[1][0], rhs.data[1][1], rhs.data[1][2],
                rhs.data[2][0], rhs.data[2][1], rhs.data[2][2]
            );
            out << buffer;
        }
        else if constexpr (N == 4) {
            std::sprintf(
                buffer, format[N - 1],
                rhs.data[0][0], rhs.data[0][1], rhs.data[0][2], rhs.data[0][3],
                rhs.data[1][0], rhs.data[1][1], rhs.data[1][2], rhs.data[1][3],
                rhs.data[2][0], rhs.data[2][1], rhs.data[2][2], rhs.data[2][3],
                rhs.data[3][0], rhs.data[3][1], rhs.data[3][2], rhs.data[3][3]
            );
            out << buffer;
        }
        else {
            out << "A " << N << " x " << N << " matrix is too large to print.";
        }
        return out;
    }
    //! Operator overloading

    //! Auxiliary functions
    static SquareMatrix Zeros() {
        return SquareMatrix{};
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
                T.data[i][j] = this->data[j][i];
            }
        }
        return T;
    }

    constexpr Float determinant() const {
        return 0;
    }

    constexpr SquareMatrix inverse() const {
        return {};
    }
    //! Auxiliary functions

private:
    Float absMax() const {
        Float abs_max{};
        for (std::size_t i = 0; i < N * N; ++i) {
            if (abs_max < std::abs(this->data_1d[i])) {
                abs_max = this->data_1d[i];
            }
        }
        return abs_max;
    }

private:
    Float* begin() { return std::begin(this->data_1d); }
    Float* end() { return std::end(this->data_1d); }
    Float const* begin() const { return std::begin(this->data_1d); }
    Float const* end() const { return std::end(this->data_1d); }

    union {
        Float data[N][N];
        Float data_1d[N * N] {};
    };
};

//! Specializations of N = 1, 2, 3, 4

//* N = 1
template <>
constexpr Float SquareMatrix<1>::determinant() const {
    return this->data[0][0];
}
template <>
constexpr SquareMatrix<1> SquareMatrix<1>::inverse() const {
    if (this->data[0][0] == Float(0)) {
        return SquareMatrix<1>{};
    }
    Float mat[1] = { Float(1) / this->data[0][0] };
    return SquareMatrix<1>{mat};
}

//* N = 2
template <>
constexpr Float SquareMatrix<2>::determinant() const {
    return crossProductDifference(
        this->data[0][0], this->data[0][1],
        this->data[1][0], this->data[1][1]
    );
}
template <>
constexpr SquareMatrix<2> SquareMatrix<2>::inverse() const {
    Float det = this->determinant();
    if (det == 0) {
        return SquareMatrix<2>{};
    }
    Float inv_det = Float(1) / det;

    Float mat[4] {
         this->data[1][1], -this->data[0][1],
        -this->data[1][0],  this->data[0][0]
    };
    mat[0] *= inv_det;
    mat[1] *= inv_det;
    mat[2] *= inv_det;
    mat[3] *= inv_det;
    
    return SquareMatrix<2>{mat};
}

//* N = 3
template <>
constexpr Float SquareMatrix<3>::determinant() const {
    Float a1 = this->data[0][0] * this->data[1][1] * this->data[2][2];
    Float a2 = this->data[0][1] * this->data[1][2] * this->data[2][0];
    Float a3 = this->data[0][2] * this->data[1][0] * this->data[2][1];

    Float b1 = this->data[0][0] * this->data[1][2] * this->data[2][1];
    Float b2 = this->data[0][1] * this->data[1][0] * this->data[2][2];
    Float b3 = this->data[0][2] * this->data[1][1] * this->data[2][0];

    return a1 + a2 + a3 - b1 - b2 - b3;
}
template <>
constexpr SquareMatrix<3> SquareMatrix<3>::inverse() const {
    Float det = this->determinant();
    if (det == 0) {
        return SquareMatrix<3>{};
    }
    Float inv_det = Float(1) / det;

    Float mat[9] {
        crossProductDifference(this->data[1][1], this->data[1][2], this->data[2][1], this->data[2][2]),
        crossProductDifference(this->data[2][1], this->data[2][2], this->data[0][1], this->data[0][2]),
        crossProductDifference(this->data[0][1], this->data[0][2], this->data[1][1], this->data[1][2]),

        crossProductDifference(this->data[2][0], this->data[2][2], this->data[1][0], this->data[1][2]),
        crossProductDifference(this->data[0][0], this->data[0][2], this->data[2][0], this->data[2][2]),
        crossProductDifference(this->data[1][0], this->data[1][2], this->data[0][0], this->data[0][2]),

        crossProductDifference(this->data[1][0], this->data[1][1], this->data[2][0], this->data[2][1]),
        crossProductDifference(this->data[2][0], this->data[2][1], this->data[0][0], this->data[0][1]),
        crossProductDifference(this->data[0][0], this->data[0][1], this->data[1][0], this->data[1][1])
    };
    for (auto&& i : mat) { i *= inv_det; }

    return SquareMatrix<3>{mat};
}

//* N = 4
template <>
constexpr Float SquareMatrix<4>::determinant() const {
    // By a little trick...
    Float M0011 = crossProductDifference(this->data[0][0], this->data[1][1], this->data[0][1], this->data[1][0]);
    Float M0012 = crossProductDifference(this->data[0][0], this->data[1][2], this->data[0][2], this->data[1][0]);
    Float M0013 = crossProductDifference(this->data[0][0], this->data[1][3], this->data[0][3], this->data[1][0]);

    Float M0112 = crossProductDifference(this->data[0][1], this->data[1][2], this->data[0][2], this->data[1][1]);
    Float M0113 = crossProductDifference(this->data[0][1], this->data[1][3], this->data[0][3], this->data[1][1]);
    Float M0213 = crossProductDifference(this->data[0][2], this->data[1][3], this->data[0][3], this->data[1][2]);

    Float M2031 = crossProductDifference(this->data[2][0], this->data[3][1], this->data[2][1], this->data[3][0]);
    Float M2032 = crossProductDifference(this->data[2][0], this->data[3][2], this->data[2][2], this->data[3][0]);
    Float M2033 = crossProductDifference(this->data[2][0], this->data[3][3], this->data[2][3], this->data[3][0]);

    Float M2132 = crossProductDifference(this->data[2][1], this->data[3][2], this->data[2][2], this->data[3][1]);
    Float M2133 = crossProductDifference(this->data[2][1], this->data[3][3], this->data[2][3], this->data[3][1]);
    Float M2233 = crossProductDifference(this->data[2][1], this->data[3][3], this->data[2][3], this->data[3][2]);

    Float result = (
        crossProductDifference(M0011, M0012, M2133, M2233) + 
        crossProductDifference(M0112, M0113, M2032, M2033) + 
        crossProductSum(M0013, M0213, M2031, M2132)
    );

    return 0;
}
template <>
constexpr SquareMatrix<4> SquareMatrix<4>::inverse() const {
    Float det = this->determinant();
    if (det == 0) {
        return SquareMatrix<4>{};
    }
    Float inv_det = Float(1) / det;

    Float mat[16] {
    };
    for (auto&& i : mat) { i *= inv_det; }

    return SquareMatrix<4>{};
}

//! Specializations of N = 1, 2, 3, 4

PBRT_NAMESPACE_END

#endif //!_PBRT_SQUARE_MATRIX_H_
