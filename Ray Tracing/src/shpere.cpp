#include <memory>

#include "hittable.hpp"
#include "sphere.hpp"
#include "vec3.hpp"

namespace RayTracing {

    Sphere::Sphere(Point3 center, float radius, std::shared_ptr<Material> mat_ptr) : center(center), radius(radius), mat_ptr(mat_ptr) {};

    bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        Vec3 oc = r.origin - this->center;
        float a = r.direction.length_squared();
        float b = dot(r.direction, oc);
        float c = oc.length_squared() - this->radius * this->radius;

        float delta = b * b - a * c;
        if (delta < 0.0f)
            return false;
        float sqrtdelta = std::sqrt(delta);

        // Find the nearest root that lies in the acceptable range.
        float root = (-b - sqrtdelta) / a;
        if (root < t_min || t_max < root) {
            root = (-b + sqrtdelta) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.hit_point = r.at(rec.t);
        Vec3 outward_normal = (rec.hit_point - this->center) / this->radius;
        rec.set_face_normal(r, outward_normal);
        getSphereUV(rec.u, rec.v, outward_normal);
        rec.mat_ptr = this->mat_ptr;

        return true;
    }

    bool Sphere::bounding_box(float t_min, float t_max, AABB& aabb) const {
        aabb = AABB(
            this->center - Vec3{ this->radius, this->radius, this->radius },
            this->center + Vec3{ this->radius, this->radius, this->radius }
        );
        return true;
    }

    bool Sphere::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
        return false;
    }

    void Sphere::getSphereUV(float& u, float& v, const Point3& p) {
        u = (std::atan2(-p[2], p[0]) + pi) / (2.0f * pi);
        v = std::acos(p[1]) / pi;
    }

}
