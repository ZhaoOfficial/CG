#include <iostream>

#include "arguments_parser.h"
#include "log.h"

namespace pbrt {
    void usage(const char *msg) {
        if (msg != nullptr) {
            fprintf(stderr, "message: %s\n\n", msg);
        }

        fprintf(stderr, R"(usage
Rendering options:
    --help          Print this help information.
    --nthreads <>
)");
    }
}
