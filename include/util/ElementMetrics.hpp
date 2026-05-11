/**
 * @file ElementMetrics.hpp
 * @brief Per-element geometric metrics used by CFL / PPW checks.
 *
 * Extracted so the same function can be reused by both SimulationFacade
 * (pure single-physics) and CoupledSimulationFacade (fluid-solid split),
 * which each run their own CheckCFL over their local mesh.
 */

#ifndef SEM_UTIL_ELEMENT_METRICS_HPP
#define SEM_UTIL_ELEMENT_METRICS_HPP

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace SEM {

/**
 * @brief Minimum Euclidean distance between adjacent GLL points inside
 *        a single element (tensor-product nearest-neighbour in the
 *        reference frame, transformed to physical space).
 *
 * Used as the conservative `h_min` in the per-element CFL criterion
 *   dt_max = cfl_factor · h_min / v_max_elem.
 *
 * Unlike the crude h/N² approximation this is exact for general curved
 * and anisotropic elements — important for the non-cubic meshes that
 * tend to show up once a user stitches a fluid half onto a solid half.
 *
 * @tparam Dim Spatial dimension (2 or 3)
 * @param mesh Parallel mesh (any mesh that satisfies mfem::ParMesh API)
 * @param e Local element index
 * @param order H1 polynomial order (so `ngll = order + 1`)
 * @return Minimum GLL neighbour distance, in physical length units.
 */
template<int Dim>
inline mfem::real_t ComputeElementMinGLLDistance(
    mfem::ParMesh& mesh, int e, int order)
{
    using mfem::real_t;

    const int ngll = order + 1;

    mfem::ElementTransformation* T = mesh.GetElementTransformation(e);

    // GLL rule (Gauss-Lobatto, not default Gauss-Legendre).
    static mfem::IntegrationRules gll_rules(0, mfem::Quadrature1D::GaussLobatto);
    const mfem::IntegrationRule& ir = gll_rules.Get(
        mesh.GetElementGeometry(e), 2 * order - 1);

    mfem::DenseMatrix pts(Dim, ir.GetNPoints());
    T->Transform(ir, pts);

    real_t min_dist_sq = std::numeric_limits<real_t>::max();

    auto dist_sq = [&](int a, int b) {
        real_t s = 0.0;
        for (int d = 0; d < Dim; ++d) {
            const real_t diff = pts(d, a) - pts(d, b);
            s += diff * diff;
        }
        return s;
    };

    if constexpr (Dim == 2) {
        for (int j = 0; j < ngll; ++j) {
            for (int i = 0; i < ngll; ++i) {
                const int idx = i + j * ngll;
                if (i < ngll - 1) {
                    min_dist_sq = std::min(min_dist_sq,
                                           dist_sq(idx, (i + 1) + j * ngll));
                }
                if (j < ngll - 1) {
                    min_dist_sq = std::min(min_dist_sq,
                                           dist_sq(idx, i + (j + 1) * ngll));
                }
            }
        }
    } else {   // Dim == 3
        const int nn = ngll * ngll;
        for (int k = 0; k < ngll; ++k) {
            for (int j = 0; j < ngll; ++j) {
                for (int i = 0; i < ngll; ++i) {
                    const int idx = i + j * ngll + k * nn;
                    if (i < ngll - 1) {
                        min_dist_sq = std::min(min_dist_sq,
                            dist_sq(idx, (i + 1) + j * ngll + k * nn));
                    }
                    if (j < ngll - 1) {
                        min_dist_sq = std::min(min_dist_sq,
                            dist_sq(idx, i + (j + 1) * ngll + k * nn));
                    }
                    if (k < ngll - 1) {
                        min_dist_sq = std::min(min_dist_sq,
                            dist_sq(idx, i + j * ngll + (k + 1) * nn));
                    }
                }
            }
        }
    }

    return std::sqrt(min_dist_sq);
}

}  // namespace SEM

#endif  // SEM_UTIL_ELEMENT_METRICS_HPP
