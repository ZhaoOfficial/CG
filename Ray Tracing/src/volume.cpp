#include "volume.hpp"

namespace RayTracing {
	ConstantMedium::ConstantMedium(std::shared_ptr<Hittable> boun_ptr, float density, std::shared_ptr<Texture> tex_ptr) : boundary(boun_ptr), neg_inv_density(-1.0f / density), phase_function(std::make_shared<Isotropic>(tex_ptr)) {}
	ConstantMedium::ConstantMedium(std::shared_ptr<Hittable> boun_ptr, float density, Color c) : boundary(boun_ptr), neg_inv_density(-1.0f / density), phase_function(std::make_shared<Isotropic>(c)) {}

	bool ConstantMedium::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
		HitRecord rec1, rec2;

		if (boundary->hit(r, -infinity, infinity, rec1) == false) {
			return false;
		}
		if (boundary->hit(r, rec1.t + 1e-4f, infinity, rec2) == false) {
			return false;
		}

		if (rec1.t < t_min) {
			rec1.t = t_min;
		}
		if (rec2.t > t_max) {
			rec2.t = t_max;
		}

		if (rec1.t >= rec2.t) {
			return false;
		}

		if (rec1.t < 0.0f) {
			rec1.t = 0.0f;
		}

		float distance_inside_boundary = (rec2.t - rec1.t) * 1.0f;
		float hit_distance = this->neg_inv_density * std::log(uniform_float(0.0f, 1.0f));

		if (hit_distance > distance_inside_boundary) {
			return false;
		}

		rec.t = rec1.t + hit_distance;
		rec.hit_point = r.at(rec.t);
		rec.normal = random_in_unit_sphere();
		rec.front_face = uniform_float(0.0f, 1.0f) < 0.5f;
		rec.mat_ptr = this->phase_function;

		return true;
	}

	bool ConstantMedium::bounding_box(float time0, float time1, AABB& aabb) const {
		return this->boundary->bounding_box(time0, time1, aabb);
	}

	bool ConstantMedium::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
		return false;
	}

}
