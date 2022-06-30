#include "common.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void PathChecker::checkNumArgs(int argc) {
    if (argc != 2) {
        std::printf("Number of arguments must be 2.\n");
        std::exit(EXIT_FAILURE);
    }
}

void PathChecker::checkPath(int argc, char **argv) {
    checkNumArgs(argc);
    std::filesystem::path file_path(argv[1]);
    if (file_path.has_extension()) {
        std::printf("Output path [%s] invalid.\n", file_path.string().c_str());
        std::exit(EXIT_FAILURE);
    }
    std::printf("Output path: %s\n", argv[1]);
}

void PathChecker::checkFilePath(int argc, char **argv, std::string_view ext) {
    checkNumArgs(argc);
    std::filesystem::path file_path(argv[1]);
    if (file_path.extension() != ext) {
        std::printf("Output path [%s] invalid.\n", file_path.string().c_str());
        std::exit(EXIT_FAILURE);
    }
    std::printf("Output path: %s\n", argv[1]);
}

Bitmap::Bitmap(std::size_t x_dim, std::size_t y_dim)
    : x_dim{x_dim}, y_dim{y_dim}
    , bitmap(x_dim * y_dim * 4) {
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.size() * sizeof(uint8_t)));
}

Bitmap::~Bitmap() {
    HANDLE_ERROR(cudaFree(dev_bitmap));
}

void Bitmap::memcpyDeviceToHost() {
    HANDLE_ERROR(cudaMemcpy(bitmap.data(), dev_bitmap, bitmap.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void Bitmap::toImage(std::string const& path) const {
    stbi_flip_vertically_on_write(1);
    stbi_write_png(path.c_str(), x_dim, y_dim, 4, bitmap.data(), 0);
    std::printf("Image [%s] output successfully!\n", path.c_str());
}

uint8_t const* Bitmap::data() const {
    return bitmap.data();
}

std::size_t Bitmap::size() const {
    return bitmap.size();
}

std::size_t Bitmap::numPixels() const {
    return x_dim * y_dim;
}
