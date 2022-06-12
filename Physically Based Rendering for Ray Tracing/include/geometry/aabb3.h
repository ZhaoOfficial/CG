#ifndef _PBRT_AABB3_H_
#define _PBRT_AABB3_H_

#include <limits>

#include "vector3.h"
#include "../common.h"

PBRT_NAMESPACE_START

template <typename T>
class Bbox3 {
public:
    //! Constructor and destructor
    Bbox3() = default;
    explicit Bbox3(Point3<T> const& point) : minimum{point}, maximum{point} {}
    Bbox3(
        Point3<T> const& p1, Point3<T> const& p2
    ) : minimum(p1.cwiseMin(p2)), maximum(p1.cwiseMax(p2)) {}
    //! Constructor and destructor

    //! Operator overload
    //* Comparation operators
    friend bool operator==(Bbox3<T> const& lhs, Bbox3<T> const& rhs) {
        return (lhs.minimum == rhs.minimum) and (lhs.maximum == rhs.maximum);
    }
    friend bool operator!=(Bbox3<T> const& lhs, Bbox3<T> const& rhs) {
        return (lhs.minimum != rhs.minimum) or (lhs.maximum != rhs.maximum);
    }
    //* Indexing operators
    Point3<T> const& operator[](std::size_t idx) const { return this->data[idx]; }
    Point3<T>& operator[](std::size_t idx) { return this->data[idx]; }
    friend std::ostream& operator<<(std::ostream& out, Bbox3<T> const& rhs) {
        out << "[" << rhs.minimum << " ~ " << rhs.maximum << "]";
        return out;
    }
    //! Operator overload

    //! Auxiliary functions
    bool empty() const { return (this->minimum.x >= this->maximum.x) or (this->minimum.y >= this->maximum.y); }
    Vector3<T> diagonal() const { return this->maximum - this->minimum; }
    T measure() const { Vector3<T> diag = this->diagonal(); return (diag.x * diag.y * diag.z); }

    Vector3<T> normCoord(Point3<T> const& point) const {
        Vector3<T> result = (point - this->minimum);
        Vector3<T> diag = this->diagonal();
        result.x /= diag.x;
        result.y /= diag.y;
        result.z /= diag.z;
        return result;
    }

    Point3<T> realCoord(Vector3<T> const& t) {
        return lerp(t, this->minimum, this->maximum);
    }

    void expand(Point3<T> const& point) {
        this->minimum = this->minimum.cwiseMin(point);
        this->maximum = this->maximum.cwiseMax(point);
    }

    Bbox3 unions(Bbox3 const& rhs) {
        auto temp = *this;
        temp.minimum = this->minimum.cwiseMin(rhs.minimum);
        temp.maximum = this->maximum.cwiseMax(rhs.maximum);
        return temp;
    }

    Bbox3 intersect(Bbox3 const& rhs) {
        auto temp = *this;
        temp.minimum = this->minimum.cwiseMax(rhs.minimum);
        temp.maximum = this->maximum.cwiseMin(rhs.maximum);
        return temp;
    }

    bool overlap(Bbox3 const& rhs) {
        return (
            (this->minimum.x <= rhs.maximum.x) and (this->minimum.y <= rhs.maximum.y) and (this->minimum.z <= rhs.maximum.z) and
            (this->maximum.x >= rhs.minimum.x) and (this->maximum.y >= rhs.minimum.y) and (this->maximum.z >= rhs.minimum.z)
        );
    }

    bool contains(Point3<T> const& point) const {
        return (
            (this->minimum.x <= point.x) and (this->minimum.y <= point.y) and (this->minimum.z <= point.z) and
            (this->maximum.x >= point.x) and (this->maximum.y >= point.y) and (this->maximum.z >= point.z)
        );
    }

    Point3<T> corner(std::size_t idx) const {
        assert(idx >= 0 and idx < 8);
        return Point3<T> {
            this->operator[](idx & 0b1).x,
            this->operator[]((idx & 0b10) >> 1).y,
            this->operator[]((idx & 0b100) >> 2).z
        };
    }
    //! Auxiliary functions

private:
    union {
        Point3<T> data[2] = {
            Point3<T>{ std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest() },
            Point3<T>{ std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max() }
        };
        struct {
            Point3<T> minimum;
            Point3<T> maximum;
        };
    };
};

using Bbox3i = Bbox3<int>;
using Bbox3f = Bbox3<Float>;

PBRT_NAMESPACE_END

#endif // !_PBRT_AABB3_H_
