#include <iostream>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <cstring>
#include <chrono>

#include "camera.hpp"
#include "hittable_list.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "utility.hpp"
#include "vec3.hpp"
#include "write_image.hpp"

RayTracing::Color rayColor(const RayTracing::Ray& r, const RayTracing::HittableList& world, int depth) {
    RayTracing::HitRecord rec;

    if (depth <= 0)
        return RayTracing::Color(0.0f, 0.0f, 0.0f);

    if (world.hit(r, 0.001f, infinity, rec)) {
        RayTracing::Color attenuation;
        RayTracing::Ray scattered;
        if (world.scatter(r, rec, attenuation, scattered)) {
            return attenuation * rayColor(scattered, world, depth - 1);
        }
        return RayTracing::Color(0.0f, 0.0f, 0.0f);
    }

    // background color
    RayTracing::Vec3 unit_direction = RayTracing::unit_vector(r.direction);
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * RayTracing::Color(1.0f, 1.0f, 1.0f) + t * RayTracing::Color(0.5f, 0.7f, 1.0f);
}

RayTracing::HittableList random_scene() {
    RayTracing::HittableList world;

    auto ground_material = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.5f, 0.5f, 0.5f));
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = uniform_float();
            RayTracing::Point3 center(a + uniform_float(), 0.2f, b + uniform_float());

            if ((center - RayTracing::Point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                std::shared_ptr<RayTracing::Material> sphere_material;

                if (choose_mat < 0.8f) {
                    // diffuse
                    auto albedo = RayTracing::random_in_unit_cube(0.0f, 1.0f) * RayTracing::random_in_unit_cube(0.0f, 1.0f);
                    sphere_material = std::make_shared<RayTracing::Lambertian>(albedo);
                    world.add(std::make_shared<RayTracing::Sphere>(center, 0.2f, sphere_material));
                } else if (choose_mat < 0.95f) {
                    // Metal
                    auto albedo = RayTracing::random_in_unit_cube(0.5f, 1.0f);
                    auto fuzz = uniform_float(0.0f, 0.5f);
                    sphere_material = std::make_shared<RayTracing::Metal>(albedo, fuzz);
                    world.add(std::make_shared<RayTracing::Sphere>(center, 0.2f, sphere_material));
                } else {
                    // glass
                    sphere_material = std::make_shared<RayTracing::Dielectric>(1.5f);
                    world.add(std::make_shared<RayTracing::Sphere>(center, 0.2f, sphere_material));
                }
            }
        }
    }

    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, 1.0f, 0.0f), 1.0f, std::make_shared<RayTracing::Dielectric>(1.5f)));
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(-4.0f, 1.0f, 0.0f), 1.0f, std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.4f, 0.2f, 0.1f))));
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(4.0f, 1.0f, 0.0f), 1.0f, std::make_shared<RayTracing::Metal>(RayTracing::Color(0.7f, 0.6f, 0.5f), 0.0f)));

    return world;
}

int main() {
    //! image setting
    const int image_width = 3840;
    const int image_height = 2160;
    const int samples_per_pixel = 1000;
    const int max_depth = 50;
    const float aspect_ratio = 16.0f / 9.0f;

    //! world
    RayTracing::HittableList world = random_scene();

    //! camera setting
    RayTracing::Point3 lookfrom(13.0f, 2.0f, 3.0f);
    RayTracing::Point3 lookat(0.0f, 0.0f, 0.0f);
    RayTracing::Vec3 vup(0.0f, 1.0f, 0.0f);
    float aperture = 0.1f;
    float dist_to_focus = 10.0f;

    RayTracing::Camera scene_camera(lookfrom, lookat, vup, aspect_ratio, 20.0f, aperture, dist_to_focus, 0.0f, 1.0f);

    std::vector<unsigned char> image_pixels(image_width * image_height * 3);
    //! render
    int row = 0;
    auto start = std::chrono::high_resolution_clock::now();

    //* (0, 0) is at left down of image
    //* The rows are written out from top to bottom.
    #pragma omp parallel for default(shared) schedule(dynamic, 5)
    for (int j = 0; j < image_height; ++j) {
        #pragma omp atomic
        ++row;
        printf("\rProcession: %4d / %4d", row, image_height);

        //* The pixels are written out in rows with pixels left to right.
        for (int i = 0; i < image_width; ++i) {
            RayTracing::Color pixel_color(0.0f, 0.0f, 0.0f);
            for (int s = 0; s < samples_per_pixel; ++s) {
                float u = (static_cast<float>(i) + gaussian_float()) / static_cast<float>(image_width - 1);
                float v = (static_cast<float>(j) + gaussian_float()) / static_cast<float>(image_height - 1);
                RayTracing::Ray r = scene_camera.get_ray(u, v);
                pixel_color += rayColor(r, world, max_depth);
            }
            pixel_color /= static_cast<float>(samples_per_pixel);
            // gamma correction
            pixel_color = RayTracing::sqrt(pixel_color);
            size_t pos = (j * image_width + i) * 3;
            image_pixels[pos + 0] = static_cast<unsigned char>(256.0f * std::clamp(pixel_color[0], 0.0f, 0.999f));
            image_pixels[pos + 1] = static_cast<unsigned char>(256.0f * std::clamp(pixel_color[1], 0.0f, 0.999f));
            image_pixels[pos + 2] = static_cast<unsigned char>(256.0f * std::clamp(pixel_color[2], 0.0f, 0.999f));
        }
    }
    // RayTracing::write_ppm(std::string("weekend.ppm"), image_pixels, image_width, image_height);
    RayTracing::write_png(std::string("weekend.png"), image_pixels, image_width, image_height);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (double)std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << duration << " s\nDone.\n";
    return 0;
}