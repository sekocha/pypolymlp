/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_gtinv_data.h"


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

GtinvData::GtinvData(){}
GtinvData::~GtinvData(){}

void GtinvData::parse(const int order, const int version){

    if (order < 1) throw std::invalid_argument("Invalid order");
    if (order > 6) throw std::invalid_argument("Invalid order");
    if (version < 1) throw std::invalid_argument("Invalid version");
    if (version > 2) throw std::invalid_argument("Invalid version");

    auto libdir = get_library_directory();
    std::filesystem::path binfile =
        std::filesystem::path(libdir) /
        ("polymlp_gtinv_data_v" + std::to_string(version) 
         + "_order" + std::to_string(order) + ".bin");

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

const vector2i32& GtinvData::get_l_array() const {
    return l_array_all;
}
const vector2d& GtinvData::get_coeffs() const {
    return coeffs_all;
}
const vector3i32& GtinvData::get_m_array() const {
    return m_array_all;
}
