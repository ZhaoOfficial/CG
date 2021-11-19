#ifndef _BOUNDING_VOLUME_HIERARCHY_HPP_
#define _BOUNDING_VOLUME_HIERARCHY_HPP_

#include <memory>
#include <vector>

#include "hittable_list.hpp"

namespace RayTracing {

    class BoundingVolumeHierarchy : public Hittable {
    public:
        BoundingVolumeHierarchy();
        BoundingVolumeHierarchy(const HittableList& list, float time0, float time1);
        BoundingVolumeHierarchy(
            const std::vector<std::shared_ptr<Hittable>>& objects,
            size_t start, size_t finish,
            float time0, float time1
        );

        virtual bool hit(
            const Ray& r, float t_min, float t_max, HitRecord& rec
        ) const override;

        virtual bool bounding_box(
            float t_min, float t_max, AABB& aabb
        ) const override;

        virtual bool scatter(
            const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered
        ) const override;
    private:
        static bool box_compare(
            const std::shared_ptr<Hittable>& a,
            const std::shared_ptr<Hittable>& b,
            int axis
        );

    public:
        std::shared_ptr<Hittable> left;
        std::shared_ptr<Hittable> right;
        AABB bvh_box;
    };

    using BVH = BoundingVolumeHierarchy;

}

#endif // !_BOUNDING_VOLUME_HIERARCHY_HPP_
