#include "box.hpp"

namespace RayTracing {
    Box::Box(const Point3& minimum, const Point3& maximum, std::shared_ptr<Material> mat_ptr) : minimum(minimum), maximum(maximum) {
        faces.add(std::make_shared<XYRectangle>(minimum[0], maximum[0], minimum[1], maximum[1], minimum[2], mat_ptr));
        faces.add(std::make_shared<XYRectangle>(minimum[0], maximum[0], minimum[1], maximum[1], maximum[2], mat_ptr));
        
        faces.add(std::make_shared<XZRectangle>(minimum[0], maximum[0], minimum[2], maximum[2], minimum[1], mat_ptr));
        faces.add(std::make_shared<XZRectangle>(minimum[0], maximum[0], minimum[2], maximum[2], maximum[1], mat_ptr));
        
        faces.add(std::make_shared<YZRectangle>(minimum[1], maximum[1], minimum[2], maximum[2], minimum[0], mat_ptr));
        faces.add(std::make_shared<YZRectangle>(minimum[1], maximum[1], minimum[2], maximum[2], maximum[0], mat_ptr));
    }

    bool Box::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        return faces.hit(r, t_min, t_max, rec);
    }

    bool Box::bounding_box(float time0, float time1, AABB& aabb) const {
        aabb = AABB(this->minimum, this->maximum);
        return true;
    }

    bool Box::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
        return false;
    }
}
