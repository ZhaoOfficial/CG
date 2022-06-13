#include "math/matrix.h"
#include "math/square_matrix.h"

PBRT_NAMESPACE_START

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
    return crossProductDifference(
        this->data[0][0], this->data[0][1],
        this->data[1][0], this->data[1][1]
    );
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
    mat[0] *= inv_det;
    mat[1] *= inv_det;
    mat[2] *= inv_det;
    mat[3] *= inv_det;
    
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
        crossProductDifference(this->data[1][1], this->data[1][2], this->data[2][1], this->data[2][2]),
        crossProductDifference(this->data[2][0], this->data[2][2], this->data[1][0], this->data[1][2]),
        crossProductDifference(this->data[1][0], this->data[1][1], this->data[2][0], this->data[2][1]),
        crossProductDifference(this->data[2][1], this->data[2][2], this->data[0][1], this->data[0][2]),
        crossProductDifference(this->data[0][0], this->data[0][2], this->data[2][0], this->data[2][2]),
        crossProductDifference(this->data[2][0], this->data[2][1], this->data[0][0], this->data[0][1]),
        crossProductDifference(this->data[0][1], this->data[0][2], this->data[1][1], this->data[1][2]),
        crossProductDifference(this->data[1][0], this->data[1][2], this->data[0][0], this->data[0][2]),
        crossProductDifference(this->data[0][0], this->data[0][1], this->data[1][0], this->data[1][1])
    };
    for (auto&& i : mat) { i *= inv_det; }

    return SquareMatrix<3>{mat};
}

//* N = 4
template <>
Float SquareMatrix<4>::determinant() const {
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
SquareMatrix<4> SquareMatrix<4>::inverse() const {
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

PBRT_NAMESPACE_END
