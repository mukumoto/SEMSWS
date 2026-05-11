/**
 * @file AdjointSource.cpp
 * @brief Adjoint source construction for FWI
 */

#include "srcrecv/AdjointSource.hpp"
#include <cmath>

namespace SEM {

AdjointSource::AdjointSource(ParFiniteElementSpace* fes, MPI_Comm comm,
                             DomainType domain,
                             const std::string& misfit_type)
    : fes_(fes), comm_(comm), domain_(domain), misfit_type_(misfit_type) {}

void AdjointSource::SetObservedData(const ObservedData& observed) {
    observed_ = &observed;
}

// =============================================================================
// Create forward receivers at observed data positions
// =============================================================================

std::unique_ptr<ReceiverArray> AdjointSource::CreateForwardReceivers(
    const ParFiniteElementSpace& fes, MPI_Comm* comm,
    DomainType domain, int nt, real_t dt) {

    MFEM_VERIFY(observed_, "ObservedData not set before CreateForwardReceivers()");

    auto receivers = std::make_unique<ReceiverArray>(fes, comm, domain);

    const auto& obs_receivers = observed_->GetReceivers();
    for (const auto& obs : obs_receivers) {
        // obs.name is "<receiver>/<channel>" (slash); use the receiver-only
        // form so HDF5 group names don't get parsed as paths.
        receivers->AddReceiver(obs.source_receiver, obs.position, obs.type);
    }

    receivers->Setup(nt, dt);
    return receivers;
}

// =============================================================================
// Compute residual and misfit (local receivers only).
//
// Each ObservedReceiver holds one physical receiver × channel with ncomp
// components as columns of `data`/`weight`. Pair observed with synthetic
// by (name, type) for 1:1 matching:
//
//   - Each synthetic ReceiverData has Data() shape (nt, ncomp_syn) where
//     ncomp_syn matches ReceiverArray::AddReceiver's rule (1 for scalar,
//     space_dim for vector channels). We require ncomp_obs == ncomp_syn.
//
//   - L2 misfit: independent per-component residual sum
//         J = 0.5 Σ_c Σ_t (obs_c - syn_c)² · w_c · dt
//
//   - Normalized-correlation misfit (vector joint form):
//         ‖s‖² = Σ_{c,t} s_c²·w·dt,  ‖o‖² = Σ_{c,t} o_c²·w·dt
//         J = 0.5 Σ_{c,t} (s_c/‖s‖ − o_c/‖o‖)² · w · dt  =  1 − NCC
//     Degenerates to the original scalar NCC when ncomp == 1, so acoustic
//     PS behavior is bitwise unchanged.
// =============================================================================

namespace {
// Build (name,type) → ReceiverData* lookup so obs can find its synthetic
// counterpart without relying on iteration-order coincidences.
std::map<std::pair<std::string, ReceiverType>, const ReceiverData*>
BuildSynMap(const std::map<ReceiverType, std::vector<ReceiverData>>& all_recv) {
    std::map<std::pair<std::string, ReceiverType>, const ReceiverData*> m;
    for (const auto& entry : all_recv) {
        for (const auto& rd : entry.second) {
            m[{rd.Name(), entry.first}] = &rd;
        }
    }
    return m;
}
}  // namespace

real_t AdjointSource::ComputeResidual(const ReceiverArray& receivers) {
    MFEM_VERIFY(observed_, "ObservedData not set before ComputeResidual()");

    const auto& obs_list = observed_->GetReceivers();
    const auto syn_map = BuildSynMap(receivers.GetAllReceivers());

    int nobs = static_cast<int>(obs_list.size());
    residuals_.resize(nobs);
    if (misfit_type_ == "normalized_correlation") synthetics_.resize(nobs);

    real_t local_misfit = 0.0;
    // Diagnostic-only: ½·Σ_t Σ_c (obs−syn)²·w·dt, accumulated in parallel
    // for ALL misfit kinds. Surfaced via L2MisfitValue() for logging.
    real_t local_l2_misfit = 0.0;

    for (int i = 0; i < nobs; ++i) {
        const auto& obs = obs_list[i];

        if (!obs.is_local) {
            residuals_[i].SetSize(0, 0);
            if (misfit_type_ == "normalized_correlation") {
                synthetics_[i].SetSize(0, 0);
            }
            continue;
        }

        // Match by physical-receiver name (no channel suffix) so it composes
        // with HDF5 group naming and CreateForwardReceivers' AddReceiver call.
        auto it = syn_map.find({obs.source_receiver, obs.type});
        MFEM_VERIFY(it != syn_map.end(),
            "AdjointSource::ComputeResidual: no synthetic receiver matching "
            << obs.source_receiver);
        const ReceiverData& syn = *it->second;
        const int nt = obs.num_samples;
        const int ncomp = obs.ncomp;

        MFEM_VERIFY(syn.Data().Height() >= nt,
            "Synthetic data has fewer samples than observed at "
            << obs.name);
        MFEM_VERIFY(syn.Data().Width() == ncomp,
            "Synthetic ncomp (" << syn.Data().Width() << ") != observed ncomp ("
            << ncomp << ") at " << obs.name);

        residuals_[i].SetSize(nt, ncomp);
        if (misfit_type_ == "normalized_correlation") {
            synthetics_[i].SetSize(nt, ncomp);
        }

        if (misfit_type_ == "l2_waveform") {
            // Per-component L2: J = 0.5 Σ_c Σ_t (obs_c − syn_c)² · w_c · dt
            for (int c = 0; c < ncomp; ++c) {
                for (int t = 0; t < nt; ++t) {
                    real_t r_t = obs.data(t, c) - syn.Data()(t, c);
                    real_t w_t = obs.weight(t, c);
                    residuals_[i](t, c) = r_t;
                    real_t contrib = 0.5 * w_t * r_t * r_t * obs.dt;
                    local_misfit += contrib;
                    local_l2_misfit += contrib;   // identical for L2 kind
                }
            }
        } else if (misfit_type_ == "normalized_correlation") {
            // Joint vector NCC: norms are sums over ALL components (c, t).
            real_t norm_d_sq = 0.0, norm_s_sq = 0.0;
            for (int c = 0; c < ncomp; ++c) {
                for (int t = 0; t < nt; ++t) {
                    real_t d_t = obs.data(t, c);
                    real_t s_t = syn.Data()(t, c);
                    real_t w_t = obs.weight(t, c);
                    real_t r_t = d_t - s_t;
                    residuals_[i](t, c) = r_t;
                    synthetics_[i](t, c) = s_t;
                    norm_d_sq += d_t * d_t * w_t * obs.dt;
                    norm_s_sq += s_t * s_t * w_t * obs.dt;
                    local_l2_misfit += 0.5 * w_t * r_t * r_t * obs.dt;
                }
            }
            if (norm_d_sq > 0.0 && norm_s_sq > 0.0) {
                real_t norm_d = std::sqrt(norm_d_sq);
                real_t norm_s = std::sqrt(norm_s_sq);
                for (int c = 0; c < ncomp; ++c) {
                    for (int t = 0; t < nt; ++t) {
                        real_t d_t = obs.data(t, c);
                        real_t s_t = syn.Data()(t, c);
                        real_t w_t = obs.weight(t, c);
                        real_t r_t = s_t / norm_s - d_t / norm_d;
                        local_misfit += 0.5 * r_t * r_t * w_t * obs.dt;
                    }
                }
            }
        } else {
            MFEM_ABORT("Unknown misfit type: " << misfit_type_
                       << ". Supported: l2_waveform, normalized_correlation");
        }
    }

    misfit_value_ = 0.0;
    l2_misfit_value_ = 0.0;
    MPI_Allreduce(&local_misfit, &misfit_value_, 1, MFEM_MPI_REAL_T, MPI_SUM, comm_);
    MPI_Allreduce(&local_l2_misfit, &l2_misfit_value_, 1, MFEM_MPI_REAL_T, MPI_SUM, comm_);

    return misfit_value_;
}

// =============================================================================
// Build adjoint sources (time-reversed weighted residual)
// =============================================================================

void AdjointSource::BuildAdjointSources(int nt, real_t dt,
                                         const MaterialField* kappa) {
    MFEM_VERIFY(observed_, "ObservedData not set before BuildAdjointSources()");
    MFEM_VERIFY(!residuals_.empty(),
        "ComputeResidual() must be called before BuildAdjointSources()");

    const auto& obs_list = observed_->GetReceivers();
    int nobs = static_cast<int>(obs_list.size());
    int obs_nt = observed_->NumSamples();

    sources_ = std::make_unique<PointSourceCollection>(fes_, comm_);

    // Phase 1: Register one adjoint source per ObservedReceiver (one per
    // physical receiver × channel). SingleForceSource accepts multi-component
    // STF and injects each column as a separate spatial-direction force, so
    // vector-channel receivers produce a proper vector adjoint force here.
    for (int i = 0; i < nobs; i++) {
        const auto& obs = obs_list[i];
        sources_->AddSingleForce(i, obs.position);
    }

    // Phase 2: Batch PointFinder
    sources_->Setup();

    // Phase 3: Set STF for local sources only.
    if (misfit_type_ == "l2_waveform") {
        // Per-component L2: dJ/ds_c(t) = −w_c(t) · (obs_c − syn_c)
        // pipeline (taper + time-operator + time-reverse + negate) converts
        // this into the adjoint-source STF.
        for (int i = 0; i < nobs; i++) {
            if (!sources_->GetSource(i)->IsLocal()) continue;

            const auto& obs = obs_list[i];
            const int ncomp = obs.ncomp;
            DenseMatrix stf_data(nt, ncomp);
            stf_data = 0.0;

            for (int c = 0; c < ncomp; ++c) {
                std::vector<real_t> w_adj(obs_nt, 0.0);
                for (int t = 0; t < obs_nt; ++t) {
                    w_adj[t] = obs.weight(t, c) * residuals_[i](t, c);
                }

                // Pipeline writes into column 0 of a scratch single-column
                // DenseMatrix; copy into column c of the final STF matrix.
                DenseMatrix stf_c(nt, 1);
                stf_c = 0.0;
                ApplyAdjointPipeline(w_adj, obs_nt, obs.dt, stf_c, nt, dt, obs.type);
                for (int t = 0; t < nt; ++t) {
                    stf_data(t, c) = stf_c(t, 0);
                }
            }
            sources_->GetSource(i)->SetSTF(SourceTimeFunction(stf_data, dt));
        }
    } else if (misfit_type_ == "normalized_correlation") {
        // Joint vector NCC: the norms and cross-correlation are sums over
        // ALL components of the physical receiver, so every component's
        // adjoint source depends on the full-vector norm.
        //
        // dJ/ds_c(t) = w(t,c) / ‖s‖ · [d_c(t)/‖d‖ − s_c(t)·cc/‖s‖²/‖d‖·‖d‖?]
        //
        // With cc = <s, d>_w, this reduces to the scalar-case expression
        // per-component (only the norms are now vector-joint).
        for (int i = 0; i < nobs; i++) {
            if (!sources_->GetSource(i)->IsLocal()) continue;

            const auto& obs = obs_list[i];
            const int ncomp = obs.ncomp;

            // Joint norms + cross-correlation over (c, t)
            real_t norm_d_sq = 0.0, norm_s_sq = 0.0, cc = 0.0;
            for (int c = 0; c < ncomp; ++c) {
                for (int t = 0; t < obs_nt; ++t) {
                    real_t d_t = obs.data(t, c);
                    real_t s_t = synthetics_[i](t, c);
                    real_t w_t = obs.weight(t, c);
                    norm_d_sq += d_t * d_t * w_t * obs.dt;
                    norm_s_sq += s_t * s_t * w_t * obs.dt;
                    cc += s_t * d_t * w_t * obs.dt;
                }
            }

            DenseMatrix stf_data(nt, ncomp);
            stf_data = 0.0;

            if (norm_d_sq > 0.0 && norm_s_sq > 0.0) {
                real_t norm_d = std::sqrt(norm_d_sq);
                real_t norm_s = std::sqrt(norm_s_sq);

                for (int c = 0; c < ncomp; ++c) {
                    std::vector<real_t> w_adj(obs_nt, 0.0);
                    for (int t = 0; t < obs_nt; ++t) {
                        real_t d_t = obs.data(t, c);
                        real_t s_t = synthetics_[i](t, c);
                        real_t w_t = obs.weight(t, c);
                        w_adj[t] = w_t / norm_s *
                            (d_t / norm_d - s_t * cc / (norm_s * norm_s * norm_d));
                    }
                    DenseMatrix stf_c(nt, 1);
                    stf_c = 0.0;
                    ApplyAdjointPipeline(w_adj, obs_nt, obs.dt, stf_c, nt, dt, obs.type);
                    for (int t = 0; t < nt; ++t) {
                        stf_data(t, c) = stf_c(t, 0);
                    }
                }
            }
            sources_->GetSource(i)->SetSTF(SourceTimeFunction(stf_data, dt));
        }
    } else {
        MFEM_ABORT("Unknown misfit type: " << misfit_type_
                   << ". Supported: l2_waveform, normalized_correlation");
    }
}

// =============================================================================
// Shared adjoint pipeline: taper -> d²/dt² (pressure only) -> time-reverse -> negate
// =============================================================================

void AdjointSource::ApplyAdjointPipeline(std::vector<real_t>& w_adj,
                                          int obs_nt, real_t dt_obs,
                                          DenseMatrix& stf_data,
                                          int nt, real_t dt,
                                          ReceiverType recv_type) {
    // Cosine taper at edges (5% each side)
    int taper_len = std::max(1, obs_nt / 20);
    for (int t = 0; t < taper_len; t++) {
        real_t w = 0.5 * (1.0 - std::cos(M_PI * t / taper_len));
        w_adj[t] *= w;
        w_adj[obs_nt - 1 - t] *= w;
    }

    // Apply the time-operator dual to (δm/δu) for each receiver-measurement type.
    // See ApplyTimeOperator's comment for the full table. The same pipeline
    // serves l2_waveform and normalized_correlation because both build
    // w_adj ≡ δJ/δm upstream; the operator here enacts the δm/δu duality.
    std::vector<real_t> processed = w_adj;
    ApplyTimeOperator(processed, dt_obs, recv_type);

    // Time-reverse and negate
    for (int t = 0; t < std::min(nt, obs_nt); t++) {
        int t_rev = obs_nt - 1 - t;
        if (t_rev >= 0 && t_rev < obs_nt) {
            stf_data(t, 0) = -processed[t_rev];
        }
    }
}

// =============================================================================
// ApplyTimeOperator (public static helper, unit-testable in isolation)
// =============================================================================
//
// Math:
//   Forward wave eq. is for displacement u. Measured quantity m relates to u
//   by m = O[u] where O is the observation operator. The misfit J[m] pulls
//   back to δJ/δu = O^*[δJ/δm] where O^* is the L² adjoint of O (integration
//   by parts on the time axis).
//
//     m            O         δm/δu     O^* on w_adj = δJ/δm
//     ---------    -------   -------   ---------------------
//     DISP u       I         I         identity
//     VEL  u̇       d/dt      d/dt      -d/dt    (one IBP)
//     ACC  ü       d²/dt²    d²/dt²    +d²/dt²  (two IBPs)
//     PS   -φ̈      -d²/dt²   -d²/dt²   +d²/dt²  (two IBPs, sign from p=-φ̈)
//
// Boundary handling: central-diff stencils are zero-padded at t=0 and t=N-1
// for the second derivative; first-derivative uses one-sided diffs there.
// Those edges are already tapered to ~0 by the cosine window upstream, so the
// stencil approximation there is harmless.

void AdjointSource::ApplyTimeOperator(std::vector<real_t>& x, real_t dt,
                                       ReceiverType recv_type) {
    const int n = static_cast<int>(x.size());
    if (n < 2) return;

    if (recv_type == ReceiverType::Pressure ||
        recv_type == ReceiverType::Acceleration) {
        const real_t inv_dt2 = 1.0 / (dt * dt);
        std::vector<real_t> out(n, 0.0);
        for (int t = 1; t < n - 1; ++t) {
            out[t] = (x[t + 1] - 2.0 * x[t] + x[t - 1]) * inv_dt2;
        }
        x.swap(out);
    } else if (recv_type == ReceiverType::Velocity) {
        const real_t inv_2dt = 0.5 / dt;
        std::vector<real_t> out(n, 0.0);
        for (int t = 1; t < n - 1; ++t) {
            out[t] = -(x[t + 1] - x[t - 1]) * inv_2dt;
        }
        // One-sided at endpoints so a non-zero edge doesn't silently drop.
        out[0]     = -(x[1] - x[0]) / dt;
        out[n - 1] = -(x[n - 1] - x[n - 2]) / dt;
        x.swap(out);
    }
    // Displacement / Gradient: identity — no-op
}

}  // namespace SEM
