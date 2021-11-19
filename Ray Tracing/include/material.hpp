#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_

#include "ray.hpp"
#include "texture.hpp"
#include "utility.hpp"
#include "vec3.hpp"

namespace RayTracing  {

    class Material {
    public:
        virtual bool scatter(
            const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
        ) const = 0;

        virtual Color emit(float u, float v, const Point3& p) const;
    };

    class Lambertian : public Material {
    public:
        Lambertian(const Color& albedo);
        Lambertian(std::shared_ptr<Texture> albedo);

        virtual bool scatter(
            const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
        ) const override;

    public:
        std::shared_ptr<Texture> albedo;
    };

    class Metal : public Material {
    public:
        Metal(const Color& albedo, float fuzz);

        virtual bool scatter(
            const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
        ) const override;

    public:
        Color albedo;
        float fuzz;
    };

    class Dielectric : public Material {
    public:
        Dielectric(float index_of_refraction);

        virtual bool scatter(
            const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
        ) const override;

    public:
        float refraction_rate;
    };

    class DiffuseLight : public Material {
    public:
        DiffuseLight(std::shared_ptr<Texture> e);
        DiffuseLight(Color c);

        virtual bool scatter(
            const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
        ) const override;

        virtual Color emit(float u, float v, const Point3& p) const;

    public:
        std::shared_ptr<Texture> emission;
    };

    class Isotropic : public Material {
    public:
        Isotropic(Color c);
        Isotropic(std::shared_ptr<Texture> a);

        virtual bool scatter(
            const Ray& r_in, const float& u, const float& v, const Vec3& hit_point, const Vec3& normal, bool front_face, Color& attenuation, Ray& scattered
        ) const override;

    public:
        std::shared_ptr<Texture> albedo;
    };

}

#endif // !_MATERIAL_HPP_