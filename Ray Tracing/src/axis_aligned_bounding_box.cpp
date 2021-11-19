#include "axis_aligned_bounding_box.hpp"

namespace RayTracing {

    AABB::AxisAlignedBoundingBox() {}
    AABB::AxisAlignedBoundingBox(const Point3& minimum, const Point3& maximum) : minimum(minimum), maximum(maximum) {}

    bool AABB::hit(const Ray& r, float t_min, float t_max) const {
        for (int i = 0; i < 3; ++i) {
            float inv_d = r.direction[i];
            float t0 = (minimum[i] - r.origin[i]) / inv_d;
            float t1 = (maximum[i] - r.origin[i]) / inv_d;

            if (inv_d < 0.0f) {
                std::swap(t0, t1);
            }

            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    AABB surrounding_box(AABB aabb0, AABB aabb1) {
        Point3 minimum(
            std::min(aabb0.minimum[0], aabb1.minimum[0]),
            std::min(aabb0.minimum[1], aabb1.minimum[1]),
            std::min(aabb0.minimum[2], aabb1.minimum[2])
        );
        Point3 maximum(
            std::max(aabb0.maximum[0], aabb1.maximum[0]),
            std::max(aabb0.maximum[1], aabb1.maximum[1]),
            std::max(aabb0.maximum[2], aabb1.maximum[2])
        );

        return AABB(minimum, maximum);
    }

}

