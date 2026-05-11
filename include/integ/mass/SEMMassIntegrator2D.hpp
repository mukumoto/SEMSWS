/**
 * @file SEMMassIntegrator2D.hpp
 * @brief MFEM BilinearFormIntegrator-based 2D SEM mass matrix integrators
 *
 * This file provides mass matrix integrators using the same pattern as
 * SEMIntegrators2D (stiffness). Key features:
 * - Inherits from BilinearFormIntegrator (PA compatible)
 * - Self-contained: stores all geometric data internally
 * - Unified coefficient API: caller passes rho, inv_kappa, damp*rho, etc.
 *
 * In SEM with GLL quadrature, mass matrix is diagonal:
 *   M_ii = coef_i · |J|_i · w_i
 *
 * Usage:
 *   SEMMassIntegrator2D mass_integ;
 *   mass_integ.AssemblePA(fes);
 *   mass_integ.SetVectorField(true);  // for elastic (2 components)
 *   mass_integ.AssembleDiagonalPA(M_diag, rho);  // pass coefficient
 */

#ifndef SEM_MASS_INTEGRATOR_2D_HPP
#define SEM_MASS_INTEGRATOR_2D_HPP

#include <mfem.hpp>

namespace SEM {


using namespace mfem;

/**
 * @class SEMMassIntegrator2D
 * @brief Unified 2D SEM mass integrator with coefficient parameter
 *
 * This integrator computes: diag += coef * detJ * weights
 *
 * The coefficient is provided by the caller, allowing the same integrator
 * to be used for:
 * - Elastic mass: coef = rho
 * - Acoustic mass: coef = 1/kappa
 * - Elastic damping: coef = damp * rho
 * - Acoustic damping: coef = damp / kappa
 */
class SEMMassIntegrator2D : public BilinearFormIntegrator {
public:
    SEMMassIntegrator2D() = default;
    ~SEMMassIntegrator2D() override = default;

    /**
     * @brief Called once during BilinearForm::Assemble()
     *
     * Computes and stores geometric data (detJ, weights, gather_map).
     */
    void AssemblePA(const FiniteElementSpace &fes) override;

    /**
     * @brief Set whether this is a vector field (2 components) or scalar
     *
     * For vector fields (elastic), uses gather_map_x_ and gather_map_y_.
     * For scalar fields (acoustic), uses gather_map_.
     *
     * @param is_vector true for elastic (2 components), false for acoustic (1 component)
     */
    void SetVectorField(bool is_vector);

    /**
     * @brief Assemble diagonal of mass matrix
     *
     * Computes: diag += coef * detJ * weights
     *
     * @param diag Output vector for diagonal mass [VSize]
     * @param coef Coefficient at GLL points [ne * ngll^2]
     */
    void AssembleDiagonalPA(Vector& diag, const Vector& coef) const;

    /**
     * @brief AddMultPA not used for mass - use AssembleDiagonalPA
     */
    void AddMultPA(const Vector &x, Vector &y) const override {
        MFEM_ABORT("SEMMassIntegrator: Use AssembleDiagonalPA for diagonal mass matrix");
    }

    /// Memory usage in bytes
    size_t MemoryUsage() const;

    // Accessors
    int Order() const { return order_; }
    int NumGLL() const { return ngll_; }
    int NumElements() const { return ne_; }

    // Template kernel methods (public for CUDA extended lambda support)
    template<int NGLL>
    void AssembleDiagonalPA_Opt(Vector& diag, const Vector& coef) const;

protected:
    // Common state
    const FiniteElementSpace* fespace_ = nullptr;
    int order_ = 0;
    int ngll_ = 0;
    int ne_ = 0;
    bool is_vector_field_ = true;  // true for elastic, false for acoustic

    // Geometric data
    Vector detJ_;           // [ne * ngll^2]
    Vector wx_, wy_;        // [ngll]
    Array<int> lex_ordering_;  // [ngll^2]

    // DOF mapping for scalar field
    Array<int> gather_map_;      // [ne * ngll^2]

    // DOF mapping for vector field
    Array<int> gather_map_x_;    // [ne * ngll^2]
    Array<int> gather_map_y_;    // [ne * ngll^2]

    // Internal methods
    void ComputeGeometry(const FiniteElementSpace& fes);
    void ComputeScalarGatherMap(const FiniteElementSpace& fes);
    void ComputeVectorGatherMap(const FiniteElementSpace& fes);
    bool HasGeometry() const { return detJ_.Size() > 0; }
};

}  // namespace SEM

#endif  // SEM_MASS_INTEGRATOR_2D_HPP
