#include <iostream>

#include "common.h"
#include "arg_parser/arg_parser.h"
#include "log/log.h"

PBRT_NAMESPACE_START

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

PBRT_NAMESPACE_END
