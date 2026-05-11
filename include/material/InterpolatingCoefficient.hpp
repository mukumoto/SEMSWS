#ifndef SEM_INTERPOLATING_COEFFICIENT_HPP
#define SEM_INTERPOLATING_COEFFICIENT_HPP

/**
 * @file InterpolatingCoefficient.hpp
 * @brief MFEM Coefficient classes for spatially varying material properties
 *
 * Provides coefficients that interpolate or lookup material values based on
 * spatial position or element attributes:
 * - InterpolatingCoefficient2D: Bilinear interpolation on regular 2D grid
 * - InterpolatingCoefficient3D: Trilinear interpolation on regular 3D grid
 * - AttributeCoefficient: Piecewise constant based on mesh element attributes
 *
 * These coefficients are designed to work with the Material classes
 * (IsotropicElasticMaterial, AcousticMaterial, etc.) for loading spatially
 * varying properties from files or per-region definitions.
 */

#include <mfem.hpp>
#include <map>

namespace SEM {

using mfem::Coefficient;
using mfem::DenseMatrix;
using mfem::DenseTensor;
using mfem::ElementTransformation;
using mfem::IntegrationPoint;
using mfem::real_t;
using mfem::Vector;

/**
 * @class InterpolatingCoefficient2D
 * @brief 2D bilinear interpolation coefficient on a regular grid
 *
 * Interpolates material values from a regular rectangular grid using bilinear
 * interpolation. The grid is defined by:
 * - Origin (x0, y0): Lower-left corner of the grid
 * - Spacing (dx, dy): Distance between grid points
 * - Dimensions (nx, ny): Number of grid points in each direction
 *
 * Data is stored in lexicographic order (x varies fastest):
 *   index = j * nx + i, where i is x-index, j is y-index
 *
 * Points outside the grid are clamped to the nearest boundary value.
 *
 * Example usage:
 * @code
 *   Vector vp_data(nx * ny);
 *   // ... fill data from file ...
 *   InterpolatingCoefficient2D vp_coef(vp_data, nx, ny, dx, dy, x0, y0);
 *   auto material = std::make_unique<IsotropicElasticMaterial>(
 *       vp_coef, vs_coef, rho_coef, fes, ir);
 * @endcode
 */
class InterpolatingCoefficient2D : public Coefficient {
public:
    /**
     * @brief Construct from data vector and grid parameters
     *
     * @param data Material values in lexicographic order [ny * nx]
     * @param nx Number of grid points in x direction
     * @param ny Number of grid points in y direction
     * @param dx Grid spacing in x direction
     * @param dy Grid spacing in y direction
     * @param x0 X coordinate of grid origin (lower-left corner)
     * @param y0 Y coordinate of grid origin (lower-left corner)
     */
    InterpolatingCoefficient2D(const Vector& data,
                               int nx, int ny,
                               real_t dx, real_t dy,
                               real_t x0, real_t y0);

    /**
     * @brief Evaluate coefficient at an integration point
     *
     * Transforms the integration point to physical coordinates, then
     * performs bilinear interpolation on the grid.
     *
     * @param T Element transformation
     * @param ip Integration point in reference coordinates
     * @return Interpolated value at the physical point
     */
    real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override;

private:
    DenseMatrix param_;  ///< Grid values stored as [nx, ny]
    int nx_, ny_;
    real_t dx_, dy_, x0_, y0_;
};


/**
 * @class InterpolatingCoefficient3D
 * @brief 3D trilinear interpolation coefficient on a regular grid
 *
 * Interpolates material values from a regular 3D grid using trilinear
 * interpolation. The grid is defined by:
 * - Origin (x0, y0, z0): Corner of the grid with minimum coordinates
 * - Spacing (dx, dy, dz): Distance between grid points
 * - Dimensions (nx, ny, nz): Number of grid points in each direction
 *
 * Data is stored in lexicographic order (x varies fastest, then y, then z):
 *   index = k * nx * ny + j * nx + i
 *
 * Points outside the grid are clamped to the nearest boundary value.
 */
class InterpolatingCoefficient3D : public Coefficient {
public:
    /**
     * @brief Construct from data vector and grid parameters
     *
     * @param data Material values in lexicographic order [nz * ny * nx]
     * @param nx Number of grid points in x direction
     * @param ny Number of grid points in y direction
     * @param nz Number of grid points in z direction
     * @param dx Grid spacing in x direction
     * @param dy Grid spacing in y direction
     * @param dz Grid spacing in z direction
     * @param x0 X coordinate of grid origin
     * @param y0 Y coordinate of grid origin
     * @param z0 Z coordinate of grid origin
     */
    InterpolatingCoefficient3D(const Vector& data,
                               int nx, int ny, int nz,
                               real_t dx, real_t dy, real_t dz,
                               real_t x0, real_t y0, real_t z0);

    /**
     * @brief Evaluate coefficient at an integration point
     *
     * @param T Element transformation
     * @param ip Integration point in reference coordinates
     * @return Interpolated value at the physical point
     */
    real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override;

private:
    DenseTensor param_;  ///< Grid values stored as [nx, ny, nz]
    int nx_, ny_, nz_;
    real_t dx_, dy_, dz_, x0_, y0_, z0_;
};


/**
 * @class AttributeCoefficient
 * @brief Piecewise constant coefficient based on mesh element attributes
 *
 * Returns different constant values based on the element's attribute.
 * This is useful for multi-region meshes where each region (identified
 * by its element attribute) has different material properties.
 *
 * Example YAML configuration:
 * @code
 *   material:
 *     format: by_attribute
 *     by_attribute:
 *       1:
 *         vp: 3000.0
 *         vs: 1732.0
 *         rho: 2500.0
 *       2:
 *         vp: 2000.0
 *         vs: 1000.0
 *         rho: 2000.0
 * @endcode
 *
 * Example usage:
 * @code
 *   std::map<int, real_t> vp_map = {{1, 3000.0}, {2, 2000.0}};
 *   AttributeCoefficient vp_coef(vp_map);
 * @endcode
 */
class AttributeCoefficient : public Coefficient {
public:
    /**
     * @brief Construct from attribute-to-value mapping
     *
     * @param attr_to_value Map from element attribute to material value
     * @param default_value Value to return for unmapped attributes (default: 0)
     */
    AttributeCoefficient(const std::map<int, real_t>& attr_to_value,
                         real_t default_value = 0.0);

    /**
     * @brief Evaluate coefficient at an integration point
     *
     * Looks up the element's attribute in the map and returns the
     * corresponding value, or the default value if not found.
     *
     * @param T Element transformation (provides element attribute)
     * @param ip Integration point (not used, but required by interface)
     * @return Material value for this element's attribute
     */
    real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override;

private:
    std::map<int, real_t> attr_to_value_;
    real_t default_value_;
};

}  // namespace SEM

#endif  // SEM_INTERPOLATING_COEFFICIENT_HPP
