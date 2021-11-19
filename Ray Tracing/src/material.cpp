#include <memory>

#include "material.hpp"
#include "ray.hpp"
#include "texture.hpp"
#include "utility.hpp"
#include "vec3.hpp"

namespace RayTracing {

    Color Material::emit(float u, float v, const Point3& p) const {
        return Color { 0.0f, 0.0f, 0.0f };
    }

    Lambertian::Lambertian(const Color& albedo) : albedo(std::make_shared<SolidColor>(albedo)) {}
    Lambertian::Lambertian(std::shared_ptr<Texture> albedo) : albedo(albedo) {}

    bool Lambertian::scatter(
        const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
    ) const {
        Vec3 scatter_direction = normal + unit_vector(random_in_unit_sphere());
        // unit_vector(random_in_unit_sphere()) may be opposite normal
        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = normal;
        scattered = Ray(hit_point, scatter_direction, r_in.time);
        attenuation = this->albedo->value(u, v, hit_point);
        return true;
    }

    Metal::Metal(const Color& albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

    bool Metal::scatter(
        const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
    ) const {
        Vec3 reflected_direction = reflect(r_in.direction, normal);
        scattered = Ray(hit_point, reflected_direction + this->fuzz * unit_vector(random_in_unit_sphere()), r_in.time);
        attenuation = this->albedo;
        return dot(scattered.direction, normal) > 0.0f;
    }

    Dielectric::Dielectric(float index_of_refraction) : refraction_rate(index_of_refraction) {}

    bool Dielectric::scatter(
        const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
    ) const {
        float refraction_ratio = front_face ? (1.0f / refraction_rate) : refraction_rate;

        Vec3 refracted = refract(r_in.direction, normal, refraction_ratio);

        scattered = Ray(hit_point, refracted, r_in.time);
        attenuation = Color(1.0f, 1.0f, 1.0f);
        return true;
    }

    DiffuseLight::DiffuseLight(std::shared_ptr<Texture> e) : emission(e) {}
    DiffuseLight::DiffuseLight(Color c) : emission(std::make_shared<SolidColor>(c)) {}

    bool DiffuseLight::scatter(
        const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
    ) const {
        return false;
    }

    Color DiffuseLight::emit(float u, float v, const Point3& p) const {
        return this->emission->value(u, v, p);
    }

    Isotropic::Isotropic(Color c) : albedo(std::make_shared<SolidColor>(c)) {}
    Isotropic::Isotropic(std::shared_ptr<Texture> a) : albedo(a) {}

    bool Isotropic::scatter(
        const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
    ) const {
        scattered = Ray(hit_point, random_in_unit_sphere(), r_in.time);
        attenuation = this->albedo->value(u, v, hit_point);
        return true;
    }
}
