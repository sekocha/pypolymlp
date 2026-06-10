/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_gtinv_data_v2.h"


std::string get_library_directory() {

#if defined(__linux__) || defined(__APPLE__)

    Dl_info info;

    if (dladdr((void*)&get_library_directory, &info) == 0) {
        throw std::runtime_error("dladdr failed");
    }

    return std::filesystem::path(info.dli_fname)
        .parent_path()
        .string();

#else

    throw std::runtime_error("unsupported platform");

#endif
}

GtinvDataVer2::GtinvDataVer2(){
}
GtinvDataVer2::~GtinvDataVer2(){}

void GtinvDataVer2::parse(const int order){

    if (order < 2) throw std::invalid_argument("Invalid order");
    if (order > 5) throw std::invalid_argument("Invalid order");

    auto libdir = get_library_directory();
    std::filesystem::path binfile =
        std::filesystem::path(libdir) /
        ("polymlp_gtinv_data_v2_order" + std::to_string(order) + ".bin");

    std::ifstream ifs(binfile, std::ios::binary);
    if (!ifs.is_open())
        throw std::runtime_error("Binary file not found.");
    if (!ifs.good())
        throw std::runtime_error("Binary file is broken.");

    char magic[4];
    ifs.read(magic, 4);

    int32_t block = read_int32(ifs);
    int32_t type1 = read_int32(ifs);
    l_array_all = read_2d_int(ifs);

    int32_t type2 = read_int32(ifs);
    coeffs_all = read_2d_double(ifs);

    int32_t type3 = read_int32(ifs);
    m_array_all = read_3d_int(ifs);
}

const vector2i32& GtinvDataVer2::get_l_array() const {
    return l_array_all;
}
const vector2d& GtinvDataVer2::get_coeffs() const {
    return coeffs_all;
}
const vector3i32& GtinvDataVer2::get_m_array() const {
    return m_array_all;
}
