#include <memory>

#include "hittable.hpp"
#include "material.hpp"
#include "moving_sphere.hpp"

namespace RayTracing {

    MovingSphere::MovingSphere() {}

    MovingSphere::MovingSphere(
        Point3 start_center, Point3 end_center,
        float start_time, float end_time,
        float radius, std::shared_ptr<Material> mat_ptr
    ) : start_center(start_center), end_center(end_center),
        start_time(start_time), end_time(end_time),
        radius(radius), mat_ptr(mat_ptr) {}

    Point3 MovingSphere::current_center(float time) const {
        if (time <= start_time) {
            return start_center;
        }
        else if (time >= end_time) {
            return end_center;
        }
        else {
            return start_center + ((time - start_time) / (end_time - start_time)) * (end_center - start_center);
        }
    }


    bool MovingSphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        Vec3 oc = r.origin - this->current_center(r.time);
        float a = r.direction.length_squared();
        float b = dot(r.direction, oc);
        float c = oc.length_squared() - this->radius * this->radius;

        float delta = b * b - a * c;
        if (delta < 0)
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
        Vec3 outward_normal = (rec.hit_point - this->current_center(r.time)) / this->radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = this->mat_ptr;

        return true;
    }

    bool MovingSphere::bounding_box(float time0, float time1, AABB& aabb) const {
        AABB aabb0(
            this->current_center(this->start_time) - Vec3{ this->radius, this->radius, this->radius },
            this->current_center(this->start_time) + Vec3{ this->radius, this->radius, this->radius }
        );
        AABB aabb1(
            this->current_center(this->end_time) - Vec3{ this->radius, this->radius, this->radius },
            this->current_center(this->end_time) + Vec3{ this->radius, this->radius, this->radius }
        );
        aabb = surrounding_box(aabb0, aabb1);
        return true;
    }

    bool MovingSphere::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
        return false;
    }
}
