#include "axis_aligned_rectangle.hpp"

namespace RayTracing {

    XYRectangle::XYRectangle() {}
    XYRectangle::XYRectangle(
        float x0, float x1, float y0, float y1, float z, std::shared_ptr<Material> mat_ptr
    ) : x0(x0), x1(x1), y0(y0), y1(y1), z(z), mat_ptr(mat_ptr) {}

    bool XYRectangle::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float t = (this->z - r.origin.z()) / r.direction.z();
        if (t < t_min || t > t_max)
            return false;

        float x = r.origin.x() + t * r.direction.x();
        float y = r.origin.y() + t * r.direction.y();
        if (x < this->x0 || x > this->x1 || y < this->y0 || y > this->y1)
            return false;

        rec.t = t;
        rec.hit_point = Vec3 { x, y, this->z };
        rec.u = (x - this->x0) / (this->x1 - this->x0);
        rec.v = (y - this->y0) / (this->y1 - this->y0);
        rec.set_face_normal(r, Vec3{ 0.0f, 0.0f, 1.0f });
        rec.mat_ptr = mat_ptr;

        return true;
    }

    bool XYRectangle::bounding_box(float time0, float time1, AABB& aabb) const {
        // The bounding box must have non-zero width in each dimension
        // so pad the Z dimension a small amount.
        aabb = AABB(Point3(this->x0, this->y0, this->z - 1e-4f), Point3(this->x1, this->y1, this->z + 1e-4f));
        return true;
    }

    bool XYRectangle::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
        return false;
    }

    XZRectangle::XZRectangle() {}
    XZRectangle::XZRectangle(
        float x0, float x1, float z0, float z1, float y, std::shared_ptr<Material> mat_ptr
    ) : x0(x0), x1(x1), z0(z0), z1(z1), y(y), mat_ptr(mat_ptr) {}

    bool XZRectangle::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float t = (this->y - r.origin.y()) / r.direction.y();
        if (t < t_min || t > t_max)
            return false;

        float x = r.origin.x() + t * r.direction.x();
        float z = r.origin.z() + t * r.direction.z();
        if (x < this->x0 || x > this->x1 || z < this->z0 || z > this->z1)
            return false;

        rec.t = t;
        rec.hit_point = Vec3 { x, this->y, z };
        rec.u = (x - this->x0) / (this->x1 - this->x0);
        rec.v = (z - this->z0) / (this->z1 - this->z0);
        rec.set_face_normal(r, Vec3{ 0.0f, 1.0f, 0.0f });
        rec.mat_ptr = mat_ptr;

        return true;
    }

    bool XZRectangle::bounding_box(float time0, float time1, AABB& aabb) const {
        // The bounding box must have non-zero width in each dimension
        // so pad the Y dimension a small amount.
        aabb = AABB(Point3(this->x0, this->y - 1e-4f, this->z0), Point3(this->x1, this->y + 1e-4f, this->z1));
        return true;
    }

    bool XZRectangle::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
        return false;
    }

    YZRectangle::YZRectangle() {}
    YZRectangle::YZRectangle(
        float y0, float y1, float z0, float z1, float x, std::shared_ptr<Material> mat_ptr
    ) : y0(y0), y1(y1), z0(z0), z1(z1), x(x), mat_ptr(mat_ptr) {}

    bool YZRectangle::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float t = (this->x - r.origin.x()) / r.direction.x();
        if (t < t_min || t > t_max)
            return false;

        float y = r.origin.y() + t * r.direction.y();
        float z = r.origin.z() + t * r.direction.z();
        if (y < this->y0 || y > this->y1 || z < this->z0 || z > this->z1)
            return false;

        rec.t = t;
        rec.hit_point = Vec3 { this->x, y, z };
        rec.u = (y - this->y0) / (this->y1 - this->y0);
        rec.v = (z - this->z0) / (this->z1 - this->z0);
        rec.set_face_normal(r, Vec3{ 1.0f, 0.0f, 0.0f });
        rec.mat_ptr = mat_ptr;

        return true;
    }

    bool YZRectangle::bounding_box(float time0, float time1, AABB& aabb) const {
        // The bounding box must have non-zero width in each dimension
        // so pad the X dimension a small amount.
        aabb = AABB(Point3(this->x - 1e-4f, this->y0, this->z0), Point3(this->x + 1e-4f, this->y1, this->z1));
        return true;
    }

    bool YZRectangle::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
        return false;
    }
    

}
