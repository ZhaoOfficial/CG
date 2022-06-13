#ifndef _PBRT_AABB2_H_
#define _PBRT_AABB2_H_

#include <limits>

#include "vector2.h"
#include "../common.h"

PBRT_NAMESPACE_START

template <typename T>
class Bbox2 {
public:
    //! Constructor and destructor
    Bbox2() = default;
    explicit Bbox2(Point2<T> const& point) : minimum{point}, maximum{point} {}
    Bbox2(
        Point2<T> const& p1, Point2<T> const& p2
    ) : minimum(p1.cwiseMin(p2)), maximum(p1.cwiseMax(p2)) {}
    //! Constructor and destructor

    //! Operator overload
    //* Comparation operators
    friend bool operator==(Bbox2<T> const& lhs, Bbox2<T> const& rhs) {
        return (lhs.minimum == rhs.minimum) and (lhs.maximum == rhs.maximum);
    }
    friend bool operator!=(Bbox2<T> const& lhs, Bbox2<T> const& rhs) {
        return (lhs.minimum != rhs.minimum) or (lhs.maximum != rhs.maximum);
    }
    //* Indexing operators
    Point2<T> const& operator[](std::size_t idx) const { return this->data[idx]; }
    Point2<T>& operator[](std::size_t idx) { return this->data[idx]; }
    friend std::ostream& operator<<(std::ostream& out, Bbox2<T> const& rhs) {
        out << "[" << rhs.minimum << " ~ " << rhs.maximum << "]";
        return out;
    }
    //! Operator overload

    //! Auxiliary functions
    bool empty() const { return (this->minimum.x >= this->maximum.x) or (this->minimum.y >= this->maximum.y); }
    Vector2<T> diagonal() const { return this->maximum - this->minimum; }
    // The measure (area, volume) of the box.
    T measure() const { Vector2<T> diag = this->diagonal(); return (diag.x * diag.y); }

    Vector2<T> normCoord(Point2<T> const& point) const {
        Vector2<T> result = (point - this->minimum);
        Vector2<T> diag = this->diagonal();
        result.x /= diag.x;
        result.y /= diag.y;
        return result;
    }

    Point2<T> realCoord(Vector2<T> const& t) {
        return lerp(t, this->minimum, this->maximum);
    }

    void expand(Point2<T> const& point) {
        this->minimum = this->minimum.cwiseMin(point);
        this->maximum = this->maximum.cwiseMax(point);
    }

    Bbox2 unions(Bbox2 const& rhs) {
        auto temp = *this;
        temp.minimum = this->minimum.cwiseMin(rhs.minimum);
        temp.maximum = this->maximum.cwiseMax(rhs.maximum);
        return temp;
    }

    Bbox2 intersect(Bbox2 const& rhs) {
        auto temp = *this;
        temp.minimum = this->minimum.cwiseMax(rhs.minimum);
        temp.maximum = this->maximum.cwiseMin(rhs.maximum);
        return temp;
    }

    bool overlap(Bbox2 const& rhs) {
        return (
            (this->minimum.x <= rhs.maximum.x) and (this->minimum.y <= rhs.maximum.y) and
            (this->maximum.x >= rhs.minimum.x) and (this->maximum.y >= rhs.minimum.y)
        );
    }

    bool contains(Point2<T> const& point) const {
        return (
            (this->minimum.x <= point.x) and (this->minimum.y <= point.y) and
            (this->maximum.x >= point.x) and (this->maximum.y >= point.y)
        );
    }

    Point2<T> corner(std::size_t idx) const {
        assert(idx >= 0 and idx < 4);
        return Point2<T> {
            this->operator[](idx & 0b1).x,
            this->operator[]((idx & 0b10) >> 1).y
        };
    }
    //! Auxiliary functions

private:
    union {
        Point2<T> data[2] = {
            Point2<T>{ std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest() },
            Point2<T>{ std::numeric_limits<T>::max(), std::numeric_limits<T>::max() }
        };
        struct {
            Point2<T> minimum;
            Point2<T> maximum;
        };
    };
};

using Bbox2i = Bbox2<int>;
using Bbox2f = Bbox2<Float>;

PBRT_NAMESPACE_END

#endif // !_PBRT_AABB2_H_
