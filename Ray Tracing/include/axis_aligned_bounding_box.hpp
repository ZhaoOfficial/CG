#ifndef _AXIS_ALIGNED_BOUNDING_BOX_HPP_
#define _AXIS_ALIGNED_BOUNDING_BOX_HPP_

#include "ray.hpp"
#include "vec3.hpp"

namespace RayTracing {

    class AxisAlignedBoundingBox {
    public:
        AxisAlignedBoundingBox();
        AxisAlignedBoundingBox(const Point3& minimum, const Point3& maximum);

        bool hit(const Ray& r, float t_min, float t_max) const;

    public:
        // minimum is one small edge (x, y, z) of the bounding box
        // maximum is one large edge (x, y, z) of the bounding box
        Point3 minimum;
        Point3 maximum;
    };

    using AABB = AxisAlignedBoundingBox;

    AABB surrounding_box(AABB box1, AABB box2);
}

#endif // !_AXIS_ALIGNED_BOUNDING_BOX_HPP_
