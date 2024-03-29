#ifndef _PBRT_SQUARE_MATRIX_H_
#define _PBRT_SQUARE_MATRIX_H_

#include <algorithm>
#include <array>
#include <functional>

#include "arithmetic.h"
#include "geometry/vector2.h"
#include "geometry/vector3.h"
#include "../common.h"

static char const* format[] = {
    "[[%-.4g]]",
    "\n[[%.4g, %.4g]\n [%.4g, %.4g]]",
    "\n[[%.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g]]",
    "\n[[%.4g, %.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g, %.4g]\n [%.4g, %.4g, %.4g, %.4g]]"
};

PBRT_NAMESPACE_START

// A N * N square matrix.
template <std::size_t N>
class SquareMatrix {
public:
    //! Constructor and destructor
    SquareMatrix() = default;
    SquareMatrix(std::array<Float, N * N> const& arr) {
        std::copy(arr.begin(), arr.end(), this->data_1d);
    }
    SquareMatrix(std::array<std::array<Float, N>, N> const& arr) {
        for (std::size_t i{}; i < N; ++i) {
            std::copy(arr[i].begin(), arr[i].end(), this->data[i]);
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
        Float inv_num = Float{1} / num;
        return this->operator*=(inv_num);
    }
    friend SquareMatrix operator+(SquareMatrix const& lhs, SquareMatrix const& rhs) {
        return SquareMatrix(lhs) += rhs;
    }
    friend SquareMatrix operator-(SquareMatrix const& lhs, SquareMatrix const& rhs) {
        return SquareMatrix(lhs) -= rhs;
    }
    friend SquareMatrix operator*(SquareMatrix const& lhs, Float rhs) {
        return SquareMatrix(lhs) *= rhs;
    }
    friend SquareMatrix operator*(Float lhs, SquareMatrix const& rhs) {
        return SquareMatrix(rhs) *= lhs;
    }
    friend SquareMatrix operator/(SquareMatrix const& lhs, Float rhs) {
        return SquareMatrix(lhs) /= rhs;
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
    // A matrix whose entities are all 0s.
    static SquareMatrix Zeros() {
        return SquareMatrix{};
    }

    // Check if this matrix is an identity matrix.
    bool isIdentity() const {
        // (N - 1) of (1, 0, 0, ...) pattern with (N) (0)s.
        for (std::size_t i = 0; i < N - 1; ++i) {
            if (data_1d[i * N + i] != Float{1}) {
                return false;
            }
            for (std::size_t j = 1; j <= N; ++j) {
                if (data_1d[i * N + i + j] != Float{0}) {
                    return false;
                }
            }
        }
        // last 1.
        if (data_1d[N * N - 1] != Float{1}) {
            return false;
        }
        return true;
    }

    // Return the transpose of this matrix.
    SquareMatrix transpose() const {
        SquareMatrix T;
        for (std::size_t i{}; i < N; ++i) {
            for (std::size_t j{}; j < N; ++j) {
                T.data[i][j] = this->data[j][i];
            }
        }
        return T;
    }

    // Calculate the determinant by cofactors.
    // Insufficient but we barely use it.
    constexpr Float determinant() const {
        SquareMatrix<N - 1> sub;
        Float det = 0;
        for (std::size_t i{}; i < N; ++i) {
            // Sub-matrix without row 0 and column i
            for (std::size_t j{}; j < N - 1; ++j)
                for (std::size_t k{}; k < N - 1; ++k)
                    sub[{j, k}] = this->data[j + 1][k < i ? k : k + 1];

            Float sign = (i & 1) ? Float{-1} : Float{1};
            det += sign * this->data[0][i] * sub.determinant();
        }
        return det;
    }

    constexpr SquareMatrix inverse() const {
        static_assert(N < 5, "Not Implemented.");
        return {};
    }
    //! Auxiliary functions

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
    if (this->data[0][0] == Float{0}) {
        return SquareMatrix<1>{};
    }
    std::array<Float, 1> arr = { Float{1} / this->data[0][0] };
    return SquareMatrix<1>{arr};
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
    Float inv_det = Float{1} / det;

    std::array<Float, 4> arr {
         this->data[1][1], -this->data[0][1],
        -this->data[1][0],  this->data[0][0]
    };
    for (auto&& i : arr) { i *= inv_det; }

    return SquareMatrix<2>{arr};
}

//* N = 3
template <>
constexpr Float SquareMatrix<3>::determinant() const {
    return volumeOfParallelepiped(
        std::array<Float, 9>{
            this->data[0][0], this->data[0][1], this->data[0][2],
            this->data[1][0], this->data[1][1], this->data[1][2],
            this->data[2][0], this->data[2][1], this->data[2][2]
        }
    );
}
template <>
constexpr SquareMatrix<3> SquareMatrix<3>::inverse() const {
    Float det = this->determinant();
    if (det == 0) {
        return SquareMatrix<3>{};
    }
    Float inv_det = Float{1} / det;

    std::array<Float, 9> arr {
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
    for (auto&& i : arr) { i *= inv_det; }

    return SquareMatrix<3>{arr};
}

//* N = 4
template <>
constexpr Float SquareMatrix<4>::determinant() const {
    // By a little trick...
    // Details are in https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    Float M0011 = crossProductDifference(this->data[0][0], this->data[0][1], this->data[1][0], this->data[1][1]);
    Float M0012 = crossProductDifference(this->data[0][0], this->data[0][2], this->data[1][0], this->data[1][2]);
    Float M0013 = crossProductDifference(this->data[0][0], this->data[0][3], this->data[1][0], this->data[1][3]);

    Float M0112 = crossProductDifference(this->data[0][1], this->data[0][2], this->data[1][1], this->data[1][2]);
    Float M0113 = crossProductDifference(this->data[0][1], this->data[0][3], this->data[1][1], this->data[1][3]);
    Float M0213 = crossProductDifference(this->data[0][2], this->data[0][3], this->data[1][2], this->data[1][3]);

    Float M2031 = crossProductDifference(this->data[2][0], this->data[2][1], this->data[3][0], this->data[3][1]);
    Float M2032 = crossProductDifference(this->data[2][0], this->data[2][2], this->data[3][0], this->data[3][2]);
    Float M2033 = crossProductDifference(this->data[2][0], this->data[2][3], this->data[3][0], this->data[3][3]);

    Float M2132 = crossProductDifference(this->data[2][1], this->data[2][2], this->data[3][1], this->data[3][2]);
    Float M2133 = crossProductDifference(this->data[2][1], this->data[2][3], this->data[3][1], this->data[3][3]);
    Float M2233 = crossProductDifference(this->data[2][2], this->data[2][3], this->data[3][2], this->data[3][3]);

    return mulAddPair(
        M0011, M2233, -M0012, M2133, M0013, M2132,
        M0112, M2033, M0113, -M2032, M0213, M2031
    );
}
template <>
constexpr SquareMatrix<4> SquareMatrix<4>::inverse() const {
    Float M0011 = crossProductDifference(this->data[0][0], this->data[0][1], this->data[1][0], this->data[1][1]);
    Float M0012 = crossProductDifference(this->data[0][0], this->data[0][2], this->data[1][0], this->data[1][2]);
    Float M0013 = crossProductDifference(this->data[0][0], this->data[0][3], this->data[1][0], this->data[1][3]);

    Float M0112 = crossProductDifference(this->data[0][1], this->data[0][2], this->data[1][1], this->data[1][2]);
    Float M0113 = crossProductDifference(this->data[0][1], this->data[0][3], this->data[1][1], this->data[1][3]);
    Float M0213 = crossProductDifference(this->data[0][2], this->data[0][3], this->data[1][2], this->data[1][3]);

    Float M2031 = crossProductDifference(this->data[2][0], this->data[2][1], this->data[3][0], this->data[3][1]);
    Float M2032 = crossProductDifference(this->data[2][0], this->data[2][2], this->data[3][0], this->data[3][2]);
    Float M2033 = crossProductDifference(this->data[2][0], this->data[2][3], this->data[3][0], this->data[3][3]);

    Float M2132 = crossProductDifference(this->data[2][1], this->data[2][2], this->data[3][1], this->data[3][2]);
    Float M2133 = crossProductDifference(this->data[2][1], this->data[2][3], this->data[3][1], this->data[3][3]);
    Float M2233 = crossProductDifference(this->data[2][2], this->data[2][3], this->data[3][2], this->data[3][3]);

    Float det = mulAddPair(
        M0011, M2233, -M0012, M2133, M0013, M2132,
        M0112, M2033, M0113, -M2032, M0213, M2031
    );

    if (det == 0) {
        return SquareMatrix<4>{};
    }
    Float inv_det = Float{1} / det;

    // The adjacent matrix.
    std::array<Float, 16> arr {
        mulAddPair( this->data[1][1], M2233, -this->data[1][2], M2133,  this->data[1][3], M2132),
        mulAddPair(-this->data[0][1], M2233,  this->data[0][2], M2133, -this->data[0][3], M2132),
        mulAddPair( this->data[3][1], M0213, -this->data[3][2], M0113,  this->data[3][3], M0112),
        mulAddPair(-this->data[2][1], M0213,  this->data[2][2], M0113, -this->data[2][3], M0112),

        mulAddPair(-this->data[1][0], M2233,  this->data[1][2], M2033, -this->data[1][3], M2032),
        mulAddPair( this->data[0][0], M2233, -this->data[0][2], M2033,  this->data[0][3], M2032),
        mulAddPair(-this->data[3][0], M0213,  this->data[3][2], M0013, -this->data[3][3], M0012),
        mulAddPair( this->data[2][0], M0213, -this->data[2][2], M0013,  this->data[2][3], M0012),

        mulAddPair( this->data[1][0], M2133, -this->data[1][1], M2033,  this->data[1][3], M2031),
        mulAddPair(-this->data[0][0], M2133,  this->data[0][1], M2033, -this->data[0][3], M2031),
        mulAddPair( this->data[3][0], M0113, -this->data[3][1], M0013,  this->data[3][3], M0011),
        mulAddPair(-this->data[2][0], M0113,  this->data[2][1], M0013, -this->data[2][3], M0011),

        mulAddPair(-this->data[1][0], M2132,  this->data[1][1], M2032, -this->data[1][2], M2031),
        mulAddPair( this->data[0][0], M2132, -this->data[0][1], M2032,  this->data[0][2], M2031),
        mulAddPair(-this->data[3][0], M0112,  this->data[3][1], M0012, -this->data[3][2], M0011),
        mulAddPair( this->data[2][0], M0112, -this->data[2][1], M0012,  this->data[2][2], M0011),
    };
    for (auto&& i : arr) { i *= inv_det; }

    return SquareMatrix<4>{arr};
}

//! Specializations of N = 1, 2, 3, 4

PBRT_NAMESPACE_END

#endif //!_PBRT_SQUARE_MATRIX_H_
