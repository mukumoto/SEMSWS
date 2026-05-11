/**
 * @file AdjointSource.hpp
 * @brief Adjoint source construction for FWI
 *
 * Computes misfit residuals and builds adjoint sources
 * as time-reversed weighted residuals injected at receiver positions.
 *
 * Supported misfit types:
 *   "l2_waveform":  J = 0.5 * Σ_r ∫ w * |obs - syn|² dt
 *   "normalized_correlation": J = Σ_r ∫ (syn/||syn|| - obs/||obs||)² * w * dt
 *
 * Uses composition: wraps existing PointSourceCollection for injection.
 */

#ifndef SEM_ADJOINT_SOURCE_HPP
#define SEM_ADJOINT_SOURCE_HPP

#include <mfem.hpp>
#include <memory>
#include "common/Types.hpp"
#include "srcrecv/ObservedData.hpp"
#include "srcrecv/Receiver.hpp"
#include "srcrecv/Source.hpp"
#include "material/MaterialField.hpp"

namespace SEM {

using mfem::real_t;
using mfem::Vector;
using mfem::DenseMatrix;
using mfem::ParFiniteElementSpace;
using mfem::ParGridFunction;

/**
 * @brief Adjoint source for FWI
 *
 * Workflow:
 * 1. SetObservedData() — load observed data
 * 2. CreateForwardReceivers() — build ReceiverArray at observed positions
 * 3. (run forward, record with ReceiverArray)
 * 4. ComputeResidual() — compute syn - obs, return misfit value
 * 5. BuildAdjointSources() — create time-reversed weighted residual sources
 * 6. GetSourceCollection() — pass to adjoint WaveOperator
 */
class AdjointSource {
public:
    AdjointSource(ParFiniteElementSpace* fes, MPI_Comm comm,
                  DomainType domain,
                  const std::string& misfit_type = "l2_waveform");

    /**
     * @brief Set observed data for this source
     */
    void SetObservedData(const ObservedData& observed);

    /**
     * @brief Create ReceiverArray at observed data positions for forward recording
     *
     * @param fes Finite element space
     * @param comm MPI communicator pointer
     * @param domain Domain type (Fluid for acoustic)
     * @param nt Number of time steps
     * @param dt Time step
     * @return ReceiverArray ready for Record() calls during forward
     */
    std::unique_ptr<ReceiverArray> CreateForwardReceivers(
        const ParFiniteElementSpace& fes, MPI_Comm* comm,
        DomainType domain, int nt, real_t dt);

    /**
     * @brief Compute residual and misfit value
     *
     * r(t) = syn(t) - obs(t)
     * J = 0.5 * Σ_r ∫ w(r,t) * |r(t)|² dt
     *
     * @param receivers Forward ReceiverArray with recorded synthetics
     * @return Misfit value J (MPI_Allreduced)
     */
    real_t ComputeResidual(const ReceiverArray& receivers);

    /**
     * @brief Build adjoint sources from time-reversed weighted residual
     *
     * Creates PointSourceCollection with time-reversed (syn-obs)*weight
     * as source time functions at each receiver position.
     *
     * For acoustic: uses SingleForceSource with shape/kappa
     *
     * @param nt Number of time steps
     * @param dt Time step
     * @param kappa Bulk modulus field (for acoustic, can be nullptr for elastic)
     */
    void BuildAdjointSources(int nt, real_t dt, const MaterialField* kappa);

    /**
     * @brief Get the adjoint PointSourceCollection for injection
     */
    PointSourceCollection* GetSourceCollection() { return sources_.get(); }

    /**
     * @brief Get last computed misfit value
     */
    real_t MisfitValue() const { return misfit_value_; }

    /**
     * @brief Diagnostic L2 misfit, computed in parallel to MisfitValue() for
     * any misfit kind. NEVER used for adjoint source seeding — exposed only
     * so the driver can log it alongside the primary misfit. For the
     * `l2_waveform` kind this equals MisfitValue() exactly.
     */
    real_t L2MisfitValue() const { return l2_misfit_value_; }

private:
    ParFiniteElementSpace* fes_;
    MPI_Comm comm_;
    DomainType domain_;
    std::string misfit_type_;

    const ObservedData* observed_ = nullptr;

    // Residual storage: [nt x 1] per local receiver (empty for non-local)
    std::vector<DenseMatrix> residuals_;

    // Synthetic data storage: [nt x 1] per local receiver
    // (needed for normalized_correlation adjoint source)
    std::vector<DenseMatrix> synthetics_;

    // Adjoint source collection
    std::unique_ptr<PointSourceCollection> sources_;

    // Shared adjoint pipeline: taper -> time-operator -> time-reverse -> negate
    void ApplyAdjointPipeline(std::vector<real_t>& w_adj, int obs_nt,
                               real_t dt_obs, DenseMatrix& stf_data,
                               int nt, real_t dt, ReceiverType recv_type);

    real_t misfit_value_ = 0.0;
    real_t l2_misfit_value_ = 0.0;   // diagnostic, never feeds adjoint

public:
    /// Apply the (δm/δu)^* time-operator in place for the given receiver
    /// measurement type. Exposed for unit testing.
    ///   Displacement / Gradient → identity
    ///   Velocity                 → -d/dt   (central, one-sided at boundaries)
    ///   Acceleration / Pressure  → +d²/dt² (3-point, zero at boundaries)
    static void ApplyTimeOperator(std::vector<real_t>& x, real_t dt,
                                  ReceiverType recv_type);
};

}  // namespace SEM

#endif  // SEM_ADJOINT_SOURCE_HPP
