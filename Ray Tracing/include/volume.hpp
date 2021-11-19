#ifndef _VOLUME_HPP_
#define _VOLUME_HPP_

#include "hittable.hpp"
#include "material.hpp"
#include "texture.hpp"

namespace RayTracing {
	
	class ConstantMedium : public Hittable {
	public:
		ConstantMedium(std::shared_ptr<Hittable> boun_ptr, float density, std::shared_ptr<Texture> tex_ptr);
		ConstantMedium(std::shared_ptr<Hittable> boun_ptr, float density, Color c);

		virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
		virtual bool bounding_box(float time0, float time1, AABB& aabb) const override;
		virtual bool scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const override;

	public:
		std::shared_ptr<Hittable> boundary;
		std::shared_ptr<Material> phase_function;
		float neg_inv_density;
	};
}

#endif // !_VOLUME_HPP_
