#include <iostream>
#include <fstream>
#include <omp.h>
#include <cstring>
#include <chrono>

#include "utility.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include "write_ppm.hpp"
#include "hittable_list.hpp"
#include "sphere.hpp"
#include "camera.hpp"
#include "material.hpp"

RayTracing::color ray_color(const RayTracing::ray& r, const RayTracing::hittable_list& world, int depth) {
    RayTracing::hit_record rec;

    if (depth <= 0)
        return RayTracing::color(0.0f, 0.0f, 0.0f);

    if (world.hit(r, 0.001f, infinity, rec)) {
        RayTracing::color attenuation;
        RayTracing::ray scattered;
        if (world.scatter(r, rec, attenuation, scattered)) {
            return attenuation * ray_color(scattered, world, depth - 1);
        }
        return RayTracing::color(0.0f, 0.0f, 0.0f);
    }

    RayTracing::vec3 unit_direction = RayTracing::unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * RayTracing::color(1.0f, 1.0f, 1.0f) + t * RayTracing::color(0.5f, 0.7f, 1.0f);
}

RayTracing::hittable_list random_scene() {
    RayTracing::hittable_list world;

    auto ground_material = std::make_shared<lambertian>(RayTracing::color(0.5f, 0.5f, 0.5f));
    world.add(std::make_shared<RayTracing::sphere>(RayTracing::point3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = uniform_float();
            RayTracing::point3 center(a + uniform_float(), 0.2f, b + uniform_float());

            if ((center - RayTracing::point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                std::shared_ptr<material> sphere_material;

                if (choose_mat < 0.8f) {
                    // diffuse
                    auto albedo = RayTracing::random_in_unit_cube(0.0f, 1.0f) * RayTracing::random_in_unit_cube(0.0f, 1.0f);
                    sphere_material = std::make_shared<lambertian>(albedo);
                    world.add(std::make_shared<RayTracing::sphere>(center, 0.2f, sphere_material));
                } else if (choose_mat < 0.95f) {
                    // metal
                    auto albedo = RayTracing::random_in_unit_cube(0.5f, 1.0f);
                    auto fuzz = uniform_float(0.0f, 0.5f);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                    world.add(std::make_shared<RayTracing::sphere>(center, 0.2f, sphere_material));
                } else {
                    // glass
                    sphere_material = std::make_shared<dielectric>(1.5f);
                    world.add(std::make_shared<RayTracing::sphere>(center, 0.2f, sphere_material));
                }
            }
        }
    }

    world.add(std::make_shared<RayTracing::sphere>(RayTracing::point3(0.0f, 1.0f, 0.0f), 1.0f, std::make_shared<dielectric>(1.5f)));
    world.add(std::make_shared<RayTracing::sphere>(RayTracing::point3(-4.0f, 1.0f, 0.0f), 1.0, std::make_shared<lambertian>(RayTracing::color(0.4f, 0.2f, 0.1f))));
    world.add(std::make_shared<RayTracing::sphere>(RayTracing::point3(4.0f, 1.0f, 0.0f), 1.0f, std::make_shared<metal>(RayTracing::color(0.7f, 0.6f, 0.5f), 0.0f)));

    return world;
}

int main() {
    //! image setting
    const int image_width = 960;
    const int image_height = 540;
    const int samples_per_pixel = 100;
    const int max_depth = 50;
    const int thread_num = 12;
    omp_set_num_threads(thread_num);
    std::vector<RayTracing::color> image_line(image_width);
    std::ofstream image("test_ppm.ppm");

    //! world
    auto world = random_scene();

    //! camera setting
    RayTracing::point3 lookfrom(13.0f, 2.0f, 3.0f);
    RayTracing::point3 lookat(0.0f, 0.0f, 0.0f);
    RayTracing::vec3 vup(0.0f, 1.0f, 0.0f);
    float aperture = 0.1f;
    float dist_to_focus = 10.0f;

    RayTracing::camera scene_camera(lookfrom, lookat, vup, 20, aperture, dist_to_focus);
    
    //! render
    auto start = std::chrono::high_resolution_clock::now();

    image << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    //* (0, 0) is at left down of image
    //* The rows are written out from top to bottom.
    for (int j = image_height - 1; j >= 0; --j) {
        std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;

        //* The pixels are written out in rows with pixels left to right.
        #pragma omp parallel for
        for (int i = 0; i < image_width; ++i) {
            RayTracing::color pixel_color(0.0f, 0.0f, 0.0f);
            for (int s = 0; s < samples_per_pixel; ++s) {
                float u = (static_cast<float>(i) + gaussian_float()) / static_cast<float>(image_width - 1);
                float v = (static_cast<float>(j) + gaussian_float()) / static_cast<float>(image_height - 1);
                RayTracing::ray r = scene_camera.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            pixel_color /= static_cast<float>(samples_per_pixel);
            // gamma adjust
            image_line[i] = sqrt(pixel_color);
        }
        write_ppm(image, image_line, image_width);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (double)std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << duration << " s\nDone.\n";
    return 0;
}