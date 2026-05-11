#include "material/InterpolatingCoefficient.hpp"
#include <cmath>
#include <algorithm>

namespace SEM {

// ============================================================================
// InterpolatingCoefficient2D
// ============================================================================

InterpolatingCoefficient2D::InterpolatingCoefficient2D(
    const Vector& data,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t x0, real_t y0)
    : nx_(nx), ny_(ny), dx_(dx), dy_(dy), x0_(x0), y0_(y0)
{
    MFEM_VERIFY(data.Size() == nx * ny,
                "Data size (" << data.Size() << ") != nx*ny (" << nx * ny << ")");

    param_.SetSize(nx, ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            param_(i, j) = data(j * nx + i);
        }
    }
}

real_t InterpolatingCoefficient2D::Eval(ElementTransformation& T,
                                         const IntegrationPoint& ip)
{
    Vector x(2);
    T.Transform(ip, x);

    x(0) -= x0_;
    x(1) -= y0_;

    // Normalized grid coordinates
    real_t fx = x(0) / dx_;
    real_t fy = x(1) / dy_;

    int i = static_cast<int>(std::floor(fx));
    int j = static_cast<int>(std::floor(fy));

    // Check if out of bounds (with small tolerance for floating-point at boundaries)
    const real_t eps = 1e-10;
    bool out_of_bounds = (fx < -eps) || (fx > nx_ - 1 + eps) ||
                         (fy < -eps) || (fy > ny_ - 1 + eps);

    if (out_of_bounds) {
        // Only warn once per coefficient instance
        static bool warned_2d = false;
        if (!warned_2d) {
            MFEM_WARNING("Point outside material grid - clamping to boundary. "
                         "Point: (" << x(0) + x0_ << ", " << x(1) + y0_ << "), "
                         "Grid: x=[" << x0_ << ", " << x0_ + (nx_-1)*dx_ << "], "
                         "y=[" << y0_ << ", " << y0_ + (ny_-1)*dy_ << "]");
            warned_2d = true;
        }
    }

    // Local interpolation weights
    real_t xd = fx - i;
    real_t yd = fy - j;

    // Clamp indices to valid range
    i = std::max(0, std::min(i, nx_ - 1));
    j = std::max(0, std::min(j, ny_ - 1));

    // Neighbor indices (also clamped)
    int i1 = std::min(i + 1, nx_ - 1);
    int j1 = std::min(j + 1, ny_ - 1);

    // Clamp interpolation weights
    xd = std::max(real_t(0), std::min(xd, real_t(1)));
    yd = std::max(real_t(0), std::min(yd, real_t(1)));

    // Bilinear interpolation
    real_t c00 = param_(i, j);
    real_t c10 = param_(i1, j);
    real_t c01 = param_(i, j1);
    real_t c11 = param_(i1, j1);

    real_t cx0 = c00 * (1 - xd) + c10 * xd;
    real_t cx1 = c01 * (1 - xd) + c11 * xd;

    return cx0 * (1 - yd) + cx1 * yd;
}


// ============================================================================
// InterpolatingCoefficient3D
// ============================================================================

InterpolatingCoefficient3D::InterpolatingCoefficient3D(
    const Vector& data,
    int nx, int ny, int nz,
    real_t dx, real_t dy, real_t dz,
    real_t x0, real_t y0, real_t z0)
    : nx_(nx), ny_(ny), nz_(nz),
      dx_(dx), dy_(dy), dz_(dz),
      x0_(x0), y0_(y0), z0_(z0)
{
    MFEM_VERIFY(data.Size() == nx * ny * nz,
                "Data size (" << data.Size() << ") != nx*ny*nz ("
                << nx * ny * nz << ")");

    param_.SetSize(nx, ny, nz);
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                param_(i, j, k) = data(k * nx * ny + j * nx + i);
            }
        }
    }
}

real_t InterpolatingCoefficient3D::Eval(ElementTransformation& T,
                                         const IntegrationPoint& ip)
{
    Vector x(3);
    T.Transform(ip, x);

    x(0) -= x0_;
    x(1) -= y0_;
    x(2) -= z0_;

    // Normalized grid coordinates
    real_t fx = x(0) / dx_;
    real_t fy = x(1) / dy_;
    real_t fz = x(2) / dz_;

    int i = static_cast<int>(std::floor(fx));
    int j = static_cast<int>(std::floor(fy));
    int k = static_cast<int>(std::floor(fz));

    // Check if out of bounds (with small tolerance for floating-point at boundaries)
    // For trilinear interpolation, valid range is [0, n-2] for index (needs i and i+1)
    // However, at exactly the boundary (i == n-1), we should not warn
    const real_t eps = 1e-10;
    bool out_of_bounds = (fx < -eps) || (fx > nx_ - 1 + eps) ||
                         (fy < -eps) || (fy > ny_ - 1 + eps) ||
                         (fz < -eps) || (fz > nz_ - 1 + eps);

    if (out_of_bounds) {
        // Only warn once per coefficient instance
        static bool warned = false;
        if (!warned) {
            MFEM_WARNING("Point outside material grid - clamping to boundary. "
                         "Element: " << T.ElementNo << ", Attr: " << T.Attribute << ", "
                         "IP: (" << ip.x << ", " << ip.y << ", " << ip.z << "), "
                         "Physical Point: (" << x(0) + x0_ << ", " << x(1) + y0_ << ", " << x(2) + z0_ << "), "
                         "Grid: x=[" << x0_ << ", " << x0_ + (nx_-1)*dx_ << "], "
                         "y=[" << y0_ << ", " << y0_ + (ny_-1)*dy_ << "], "
                         "z=[" << z0_ << ", " << z0_ + (nz_-1)*dz_ << "]");
            warned = true;
        }
    }

    // Local interpolation weights
    real_t xd = fx - i;
    real_t yd = fy - j;
    real_t zd = fz - k;

    // Clamp indices to valid range
    i = std::max(0, std::min(i, nx_ - 1));
    j = std::max(0, std::min(j, ny_ - 1));
    k = std::max(0, std::min(k, nz_ - 1));

    // Neighbor indices (clamped)
    int i1 = std::min(i + 1, nx_ - 1);
    int j1 = std::min(j + 1, ny_ - 1);
    int k1 = std::min(k + 1, nz_ - 1);

    // Clamp interpolation weights
    xd = std::max(real_t(0), std::min(xd, real_t(1)));
    yd = std::max(real_t(0), std::min(yd, real_t(1)));
    zd = std::max(real_t(0), std::min(zd, real_t(1)));

    // Get corner values
    real_t c000 = param_(i, j, k);
    real_t c100 = param_(i1, j, k);
    real_t c010 = param_(i, j1, k);
    real_t c110 = param_(i1, j1, k);
    real_t c001 = param_(i, j, k1);
    real_t c101 = param_(i1, j, k1);
    real_t c011 = param_(i, j1, k1);
    real_t c111 = param_(i1, j1, k1);

    // Trilinear interpolation
    // Interpolate along x
    real_t c00 = c000 * (1 - xd) + c100 * xd;
    real_t c10 = c010 * (1 - xd) + c110 * xd;
    real_t c01 = c001 * (1 - xd) + c101 * xd;
    real_t c11 = c011 * (1 - xd) + c111 * xd;

    // Interpolate along y
    real_t c0 = c00 * (1 - yd) + c10 * yd;
    real_t c1 = c01 * (1 - yd) + c11 * yd;

    // Interpolate along z
    return c0 * (1 - zd) + c1 * zd;
}


// ============================================================================
// AttributeCoefficient
// ============================================================================

AttributeCoefficient::AttributeCoefficient(
    const std::map<int, real_t>& attr_to_value,
    real_t default_value)
    : attr_to_value_(attr_to_value), default_value_(default_value)
{
}

real_t AttributeCoefficient::Eval(ElementTransformation& T,
                                   const IntegrationPoint& /*ip*/)
{
    int attr = T.Attribute;
    auto it = attr_to_value_.find(attr);
    if (it != attr_to_value_.end()) {
        return it->second;
    }
    return default_value_;
}

}  // namespace SEM
