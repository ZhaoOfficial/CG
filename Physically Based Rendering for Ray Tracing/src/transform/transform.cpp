#include "common.h"
#include "transform/transform.h"

PBRT_NAMESPACE_START

bool Transform::operator==(Transform const& rhs) const { return (this->mat == rhs.mat) and (this->inv_mat == rhs.inv_mat); }
bool Transform::operator!=(Transform const& rhs) const { return (this->mat != rhs.mat) or (this->inv_mat != rhs.inv_mat); }
bool Transform::operator<(Transform const& rhs) const { return this->mat < rhs.mat; }

SquareMatrix<4> const& Transform::getMatrix() const { return this->mat; };
SquareMatrix<4> const& Transform::getInverseMatrix() const { return this->inv_mat; };

PBRT_NAMESPACE_END
