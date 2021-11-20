#include "hittable.hpp"

namespace RayTracing {
	
	Translate::Translate(std::shared_ptr<Hittable> hit_ptr, const Vec3& displacement) : hit_ptr(hit_ptr), displacement(displacement) {}

	bool Translate::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
		Ray translated_ray(r.origin - this->displacement, r.direction, r.time);
		if (this->hit_ptr->hit(translated_ray, t_min, t_max, rec) == false) {
			return false;
		}

		rec.hit_point += this->displacement;
		rec.set_face_normal(translated_ray, rec.normal);
		return true;
	}

	bool Translate::bounding_box(float time0, float time1, AABB& aabb) const {
		if (this->hit_ptr->bounding_box(time0, time1, aabb) == false) {
			return false;
		}

		aabb = AABB(
			aabb.minimum + this->displacement,
			aabb.maximum + this->displacement
		);
		return true;
	}

	bool Translate::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
		return false;
	}

	YRotate::YRotate(std::shared_ptr<Hittable> hit_ptr, float angle) : hit_ptr(hit_ptr) {
		float theta = degrees_to_radians(angle);
		this->sin_theta = std::sin(theta);
		this->cos_theta = std::cos(theta);
		this->has_box = this->hit_ptr->bounding_box(0.0f, 1.0f, this->aabb);

		float x = this->aabb.maximum.x();
		float z = this->aabb.maximum.z();
		float x_new = this->cos_theta * x + this->sin_theta * z;
		float z_new = -this->sin_theta * x + this->cos_theta * z;

		Point3 minimum(x_new, this->aabb.minimum.y(), z_new);
		Point3 maximum(x_new, this->aabb.maximum.y(), z_new);

		x = this->aabb.minimum.x();
		z = this->aabb.maximum.z();
		x_new = this->cos_theta * x + this->sin_theta * z;
		z_new = -this->sin_theta * x + this->cos_theta * z;
		minimum[0] = std::min(minimum[0], x_new);
		minimum[2] = std::min(minimum[2], z_new);
		maximum[0] = std::max(maximum[0], x_new);
		maximum[2] = std::max(maximum[2], z_new);

		x = this->aabb.maximum.x();
		z = this->aabb.minimum.z();
		x_new = this->cos_theta * x + this->sin_theta * z;
		z_new = -this->sin_theta * x + this->cos_theta * z;
		minimum[0] = std::min(minimum[0], x_new);
		minimum[2] = std::min(minimum[2], z_new);
		maximum[0] = std::max(maximum[0], x_new);
		maximum[2] = std::max(maximum[2], z_new);

		x = this->aabb.minimum.x();
		z = this->aabb.minimum.z();
		x_new = this->cos_theta * x + this->sin_theta * z;
		z_new = -this->sin_theta * x + this->cos_theta * z;
		minimum[0] = std::min(minimum[0], x_new);
		minimum[2] = std::min(minimum[2], z_new);
		maximum[0] = std::max(maximum[0], x_new);
		maximum[2] = std::max(maximum[2], z_new);

		this->aabb = AABB(minimum, maximum);
	}

	bool YRotate::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
		Vec3 origin = r.origin;
		Vec3 direction = r.direction;

		origin[0] = this->cos_theta * r.origin[0] - this->sin_theta * r.origin[2];
		origin[2] = this->sin_theta * r.origin[0] + this->cos_theta * r.origin[2];

		direction[0] = this->cos_theta * r.direction[0] - this->sin_theta * r.direction[2];
		direction[2] = this->sin_theta * r.direction[0] + this->cos_theta * r.direction[2];

		Ray rotated_ray(origin, direction, r.time);

		if (this->hit_ptr->hit(rotated_ray, t_min, t_max, rec) == false) {
			return false;
		}

		Vec3 hit_point = rec.hit_point;
		Vec3 normal = rec.normal;

		hit_point[0] = this->cos_theta * rec.hit_point[0] + this->sin_theta * rec.hit_point[2];
		hit_point[2] = -this->sin_theta * rec.hit_point[0] + this->cos_theta * rec.hit_point[2];

		normal[0] = this->cos_theta * rec.normal[0] + this->sin_theta * rec.normal[2];
		normal[2] = -this->sin_theta * rec.normal[0] + this->cos_theta * rec.normal[2];

		rec.hit_point = hit_point;
		rec.set_face_normal(rotated_ray, normal);

		return true;
	}

	bool YRotate::bounding_box(float time0, float time1, AABB& aabb) const {
		aabb = this->aabb;
		return has_box;
	}

	bool YRotate::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
		return false;
	}


}
