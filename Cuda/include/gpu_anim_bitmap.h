#ifndef _GPU_ANIM_BITMAP_
#define _GPU_ANIM_BITMAP_

#include <functional>

#include "anim_bitmap.h"

class GPUAnimBitmap: public AnimBitmap {
public:
    cudaGraphicsResource *cuda_resource{};

public:
    GPUAnimBitmap(int width, int height, std::string const& title, void *data);
    virtual ~GPUAnimBitmap() = default;

    void animate(std::function<void(uchar4*, int, int, int)> render_func);
};

#endif // !_GPU_ANIM_BITMAP_
