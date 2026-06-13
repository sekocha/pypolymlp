/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_gtinv_binary.h"


int32_t read_int32(std::ifstream &ifs) {
    uint8_t b[4];
    ifs.read((char*)b, 4);

    return (int32_t)(
        (uint32_t)b[0] |
        ((uint32_t)b[1] << 8) |
        ((uint32_t)b[2] << 16) |
        ((uint32_t)b[3] << 24)
    );
}

double read_double(std::ifstream &ifs) {
    uint8_t b[8];
    ifs.read((char*)b, 8);

    uint64_t u =
        (uint64_t)b[0] |
        ((uint64_t)b[1] << 8) |
        ((uint64_t)b[2] << 16) |
        ((uint64_t)b[3] << 24) |
        ((uint64_t)b[4] << 32) |
        ((uint64_t)b[5] << 40) |
        ((uint64_t)b[6] << 48) |
        ((uint64_t)b[7] << 56);

    double x;
    std::memcpy(&x, &u, sizeof(double));
    return x;
}

vector2i32 read_2d_int(std::ifstream &ifs) {
    int32_t n = read_int32(ifs);

    vector2i32 v(n);
    for (int i = 0; i < n; i++) {
        int32_t m = read_int32(ifs);
        v[i].resize(m);
        for (int j = 0; j < m; j++) {
            v[i][j] = read_int32(ifs);
        }
    }
    return v;
}


vector3i32 read_3d_int(std::ifstream &ifs) {
    int32_t A = read_int32(ifs);

    vector3i32 v(A);
    for (int i = 0; i < A; i++) {
        int32_t B = read_int32(ifs);

        v[i].resize(B);
        for (int j = 0; j < B; j++) {
            int32_t C = read_int32(ifs);

            v[i][j].resize(C);
            for (int k = 0; k < C; k++) {
                v[i][j][k] = read_int32(ifs);
            }
        }
    }
    return v;
}

vector2d read_2d_double(std::ifstream &ifs) {
    int32_t n = read_int32(ifs);

    vector2d v(n);
    for (int i = 0; i < n; i++) {
        int32_t m = read_int32(ifs);
        v[i].resize(m);

        for (int j = 0; j < m; j++) {
            v[i][j] = read_double(ifs);
        }
    }
    return v;
}
