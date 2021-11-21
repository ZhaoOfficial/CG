#include <iostream>
#include <algorithm>
#include <omp.h>
#include <cstring>
#include <chrono>

#include "axis_aligned_bounding_box.hpp"
#include "axis_aligned_rectangle.hpp"
#include "bounding_volume_hierarchy.hpp"
#include "box.hpp"
#include "camera.hpp"
#include "hittable_list.hpp"
#include "material.hpp"
#include "moving_sphere.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "utility.hpp"
#include "vec3.hpp"
#include "volume.hpp"
#include "write_image.hpp"

RayTracing::Color rayColor(const RayTracing::Ray& r, const RayTracing::Color& background, const RayTracing::HittableList& world, int depth) {
    RayTracing::HitRecord rec;

    if (depth <= 0)
        return RayTracing::Color(0.0f, 0.0f, 0.0f);

    if (world.hit(r, 0.001f, infinity, rec)) {
        RayTracing::Ray scattered;
        RayTracing::Color attenuation;
        RayTracing::Color emission = rec.mat_ptr->emit(rec.u, rec.v, rec.hit_point);

        if (world.scatter(r, rec, attenuation, scattered) == false) {
            return emission;
        }
        return emission + attenuation * rayColor(scattered, background, world, depth - 1);
    }

    // background Color
    return background;

    // RayTracing::Vec3 unit_direction = RayTracing::unit_vector(r.direction);
    // float t = 0.5f * (unit_direction.y() + 1.0f);
    // return (1.0f - t) * RayTracing::Color(1.0f, 1.0f, 1.0f) + t * RayTracing::Color(0.5f, 0.7f, 1.0f);
}

RayTracing::HittableList randomScene() {
    RayTracing::HittableList world;

    auto checker = std::make_shared<RayTracing::CheckerTexture>(RayTracing::Color{ 0.1f, 0.1f, 0.1f }, RayTracing::Color{ 0.9f, 0.9f, 0.9f });
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, -1000.0f, 0.0f), 1000.0f, std::make_shared<RayTracing::Lambertian>(checker)));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = uniform_float();
            RayTracing::Point3 center(a + uniform_float(), 0.2f, b + uniform_float());

            if ((center - RayTracing::Point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                std::shared_ptr<RayTracing::Material> sphere_material;

                if (choose_mat < 0.8f) {
                    // diffuse
                    auto albedo = RayTracing::random_in_unit_cube(0.0f, 1.0f) * RayTracing::random_in_unit_cube(0.0f, 1.0f);
                    sphere_material = std::make_shared<RayTracing::Lambertian>(albedo);
                    
                    RayTracing::Point3 end_center = center + RayTracing::Vec3 { 0.0f, uniform_float(0.0f, 0.5f), 0.0f };

                    world.add(
                        std::make_shared<RayTracing::MovingSphere>(
                            center, end_center, 0.0f, 1.0f, 0.2f, sphere_material
                        )
                    );
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

RayTracing::HittableList twoSpheres() {
    RayTracing::HittableList world;

    auto checker = std::make_shared<RayTracing::CheckerTexture>(RayTracing::Color{ 0.1f, 0.1f, 0.1f }, RayTracing::Color{ 0.9f, 0.9f, 0.9f });

    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, -10.0f, 0.0f), 10.0f, std::make_shared<RayTracing::Lambertian>(checker)));
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f,  10.0f, 0.0f), 10.0f, std::make_shared<RayTracing::Lambertian>(checker)));

    return world;
}

RayTracing::HittableList twoPerlin() {
    RayTracing::HittableList world;

    auto perline_texture = std::make_shared<RayTracing::NoiseTexture>(4.0f);

    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, -1000.0f, 0.0f), 1000.0f, std::make_shared<RayTracing::Lambertian>(perline_texture)));
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, 2.0f, 0.0f), 2.0f, std::make_shared<RayTracing::Lambertian>(perline_texture)));

    return world;
}

RayTracing::HittableList earth() {
    RayTracing::HittableList world;

    auto earth_texture = std::make_shared<RayTracing::ImageTexture>("../image/earthmap.jpg");
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, 0.0f, 0.0f), 2.0f, std::make_shared<RayTracing::Lambertian>(earth_texture)));

    return world;
}

RayTracing::HittableList areaLight() {
    RayTracing::HittableList world;

    auto perline_texture = std::make_shared<RayTracing::NoiseTexture>(4.5f);

    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, -1000.0f, 0.0f), 1000.0f, std::make_shared<RayTracing::Lambertian>(perline_texture)));
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, 2.0f, 0.0f), 2.0f, std::make_shared<RayTracing::Lambertian>(perline_texture)));

    auto diffuse_light = std::make_shared<RayTracing::DiffuseLight>(RayTracing::Color(4.0f, 4.0f, 4.0f));
    world.add(std::make_shared<RayTracing::XYRectangle>(3.0f, 5.0f, 1.0f, 3.0f, -2.0f, diffuse_light));
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, 7.0f, 0.0f), 2.0f, diffuse_light));

    return world;
}

RayTracing::HittableList cornellBox() {
    RayTracing::HittableList world;

    auto red   = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.65f, 0.05f, 0.05f));
    auto white = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.73f, 0.73f, 0.73f));
    auto green = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.12f, 0.45f, 0.15f));
    auto light = std::make_shared<RayTracing::DiffuseLight>(RayTracing::Color(15.0f, 15.0f, 15.0f));

    world.add(std::make_shared<RayTracing::YZRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, green));
    world.add(std::make_shared<RayTracing::YZRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, red));
    world.add(std::make_shared<RayTracing::XZRectangle>(213.0f, 343.0f, 227.0f, 332.0f, 554.0f, light));
    world.add(std::make_shared<RayTracing::XZRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, white));
    world.add(std::make_shared<RayTracing::XZRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white));
    world.add(std::make_shared<RayTracing::XYRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white));

    std::shared_ptr<RayTracing::Hittable> box1 = std::make_shared<RayTracing::Box>(RayTracing::Point3(0.0f, 0.0f, 0.0f), RayTracing::Point3(165.0f, 330.0f, 165.0f), white);
    box1 = std::make_shared<RayTracing::YRotate>(box1, 15.0f);
    box1 = std::make_shared<RayTracing::Translate>(box1, RayTracing::Vec3(265.0f, 0.0f, 295.0f));
    world.add(box1);

    std::shared_ptr<RayTracing::Hittable> box2 = std::make_shared<RayTracing::Box>(RayTracing::Point3(0.0f, 0.0f, 0.0f), RayTracing::Point3(165.0f, 165.0f, 165.0f), white);
    box2 = std::make_shared<RayTracing::YRotate>(box2, -18.0f);
    box2 = std::make_shared<RayTracing::Translate>(box2, RayTracing::Vec3(130.0f, 0.0f, 65.0f));
    world.add(box2);

    return world;
}

RayTracing::HittableList cornellBoxVolume() {
    RayTracing::HittableList world;

    auto red = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.65f, 0.05f, 0.05f));
    auto white = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.73f, 0.73f, 0.73f));
    auto green = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.12f, 0.45f, 0.15f));
    auto light = std::make_shared<RayTracing::DiffuseLight>(RayTracing::Color(7.0f, 7.0f, 7.0f));

    world.add(std::make_shared<RayTracing::YZRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, green));
    world.add(std::make_shared<RayTracing::YZRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, red));
    world.add(std::make_shared<RayTracing::XZRectangle>(113.0f, 443.0f, 127.0f, 432.0f, 554.0f, light));
    world.add(std::make_shared<RayTracing::XZRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, white));
    world.add(std::make_shared<RayTracing::XZRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white));
    world.add(std::make_shared<RayTracing::XYRectangle>(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white));

    std::shared_ptr<RayTracing::Hittable> box1 = std::make_shared<RayTracing::Box>(RayTracing::Point3(0.0f, 0.0f, 0.0f), RayTracing::Point3(165.0f, 330.0f, 165.0f), white);
    box1 = std::make_shared<RayTracing::YRotate>(box1, 15.0f);
    box1 = std::make_shared<RayTracing::Translate>(box1, RayTracing::Vec3(265.0f, 0.0f, 295.0f));

    std::shared_ptr<RayTracing::Hittable> box2 = std::make_shared<RayTracing::Box>(RayTracing::Point3(0.0f, 0.0f, 0.0f), RayTracing::Point3(165.0f, 165.0f, 165.0f), white);
    box2 = std::make_shared<RayTracing::YRotate>(box2, -18.0f);
    box2 = std::make_shared<RayTracing::Translate>(box2, RayTracing::Vec3(130.0f, 0.0f, 65.0f));

    world.add(std::make_shared<RayTracing::ConstantMedium>(box1, 0.01f, RayTracing::Color(0.0f, 0.0f, 0.0f)));
    world.add(std::make_shared<RayTracing::ConstantMedium>(box2, 0.01f, RayTracing::Color(1.0f, 1.0f, 1.0f)));

    return world;
}

RayTracing::HittableList finalScene() {
    RayTracing::HittableList boxes1;
    auto ground = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.48f, 0.83f, 0.53f));

    const int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            float w = 100.0f;
            float x0 = -1000.0f + i * w;
            float z0 = -1000.0f + j * w;
            float y0 = 0.0f;
            float x1 = x0 + w;
            float y1 = uniform_float(1.0f, 101.0f);
            float z1 = z0 + w;

            boxes1.add(std::make_shared<RayTracing::Box>(RayTracing::Point3(x0, y0, z0), RayTracing::Point3(x1, y1, z1), ground));
        }
    }

    RayTracing::HittableList world;

    world.add(std::make_shared<RayTracing::BVH>(boxes1, 0.0f, 1.0f));

    auto light = std::make_shared<RayTracing::DiffuseLight>(RayTracing::Color(7.0f, 7.0f, 7.0f));
    world.add(std::make_shared<RayTracing::XZRectangle>(123.0f, 423.0f, 147.0f, 412.0f, 554.0f, light));

    auto center1 = RayTracing::Point3(400.0f, 400.0f, 200.0f);
    auto center2 = center1 + RayTracing::Vec3(30.0f, 0.0f, 0.0f);
    auto moving_sphere_material = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.7f, 0.3f, 0.1f));
    world.add(std::make_shared<RayTracing::MovingSphere>(center1, center2, 0.0f, 1.0f, 50.0f, moving_sphere_material));

    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(260.0f, 150.0f, 45.0f), 50.0f, std::make_shared<RayTracing::Dielectric>(1.5f)));
    world.add(std::make_shared<RayTracing::Sphere>(
        RayTracing::Point3(0.0f, 150.0f, 145.0f), 50.0f, std::make_shared<RayTracing::Metal>(RayTracing::Color(0.8f, 0.8f, 0.9f), 1.0f)
    ));

    auto boundary = std::make_shared<RayTracing::Sphere>(RayTracing::Point3(360.0f, 150.0f, 145.0f), 70.0f, std::make_shared<RayTracing::Dielectric>(1.5f));
    world.add(boundary);
    world.add(std::make_shared<RayTracing::ConstantMedium>(boundary, 0.2f, RayTracing::Color(0.2f, 0.4f, 0.9f)));
    boundary = std::make_shared<RayTracing::Sphere>(RayTracing::Point3(0.0f, 0.0f, 0.0f), 5000.0f, std::make_shared<RayTracing::Dielectric>(1.5f));
    world.add(std::make_shared<RayTracing::ConstantMedium>(boundary, 1e-4f, RayTracing::Color(1.0f, 1.0f, 1.0f)));

    auto emat = std::make_shared<RayTracing::Lambertian>(std::make_shared<RayTracing::ImageTexture>("../image/earthmap.jpg"));
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(400.0f, 200.0f, 400.0f), 100.0f, emat));
    auto pertext = std::make_shared<RayTracing::NoiseTexture>(0.1f);
    world.add(std::make_shared<RayTracing::Sphere>(RayTracing::Point3(220.0f, 280.0f, 300.0f), 80.0f, std::make_shared<RayTracing::Lambertian>(pertext)));

    RayTracing::HittableList boxes2;
    auto white = std::make_shared<RayTracing::Lambertian>(RayTracing::Color(0.73f, 0.73f, 0.73f));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(std::make_shared<RayTracing::Sphere>(RayTracing::random_in_unit_cube() * 165.0f, 10.0f, white));
    }

    world.add(
        std::make_shared<RayTracing::Translate>(
            std::make_shared<RayTracing::YRotate>(std::make_shared<RayTracing::BVH>(boxes2, 0.0f, 1.0f), 15.0f),
            RayTracing::Vec3(-100.0f, 270.0f, 395.0f)
        )
    );

    return world;
}

int main() {

    //! image setting
    int image_width = 2160;
    int image_height = 2160;
    int samples_per_pixel = 1000;
    int max_depth = 50;

    //! world
    RayTracing::HittableList world;

    //! camera setting
    RayTracing::Point3 lookfrom(13.0f, 2.0f, 3.0f);
    RayTracing::Point3 lookat(0.0f, 0.0f, 0.0f);
    RayTracing::Vec3 vup(0.0f, 1.0f, 0.0f);
    float aspect_ratio = 16.0f / 9.0f;
    float vertical_fov = 20.0f;
    float aperture = 0.0f;
    float dist_to_focus = 10.0f;
    float start_time = 0.0f;
    float end_time = 1.0f;
    RayTracing::Color background(0.7f, 0.8f, 1.0f);

    switch (7) {
        case 1:
            world = randomScene();
            aperture = 0.1f;
            break;
        case 2:
            world = twoSpheres();
            break;
        case 3:
            world = twoPerlin();
            break;
        case 4:
            world = earth();
            break;
        case 5:
            world = areaLight();
            lookfrom = RayTracing::Color(26.0f, 3.0f, 6.0f);
            lookat= RayTracing::Color(0.0f, 2.0f, 0.0f);
            background = RayTracing::Color(0.0f, 0.0f, 0.0f);
            break;
        case 6:
            world = cornellBox();
            aspect_ratio = 1.0f;
            image_width = static_cast<int>(aspect_ratio * static_cast<float>(image_height));
            background = RayTracing::Color(0.0f, 0.0f, 0.0f);
            lookfrom = RayTracing::Point3(278.0f, 278.0f, -800.0f);
            lookat = RayTracing::Point3(278.0f, 278.0f, 0.0f);
            vertical_fov = 40.0f;
            break;
        case 7:
            world = cornellBoxVolume();
            aspect_ratio = 1.0f;
            image_width = static_cast<int>(aspect_ratio * static_cast<float>(image_height));
            background = RayTracing::Color(0.0f, 0.0f, 0.0f);
            lookfrom = RayTracing::Point3(278.0f, 278.0f, -800.0f);
            lookat = RayTracing::Point3(278.0f, 278.0f, 0.0f);
            vertical_fov = 40.0f;
            break;
        case 8:
            world = finalScene();
            aspect_ratio = 1.0f;
            image_width = static_cast<int>(aspect_ratio * static_cast<float>(image_height));
            background = RayTracing::Color(0.0f, 0.0f, 0.0f);
            lookfrom = RayTracing::Point3(478.0f, 278.0f, -600.0f);
            lookat = RayTracing::Point3(278.0f, 278.0f, 0.0f);
            vertical_fov = 40.0f;
            break;
        default:
        case 9:
            background = RayTracing::Color(0.0f, 0.0f, 0.0f);
            break;
    }

    RayTracing::Camera scene_camera(
        lookfrom, lookat, vup, aspect_ratio,
        vertical_fov, aperture, dist_to_focus,
        start_time, end_time
    );
    
    //! render
    std::vector<unsigned char> image_pixels(image_width * image_height * 3);

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
                pixel_color += rayColor(r, background, world, max_depth);
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
    // RayTracing::write_ppm(std::string("week.ppm"), image_pixels, image_width, image_height);
    RayTracing::write_png(std::string("week.png"), image_pixels, image_width, image_height);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    printf("\nDone: %ld s\n", duration);
    return 0;
}