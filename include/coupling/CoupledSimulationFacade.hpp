// Fluid-solid coupled simulation driver. Owns a parent ParMesh plus two
// ParSubMeshes (fluid, solid) with their own SimulationComponents, Materials,
// operators, time integrators, and a FluidSolidInterface that exchanges
// phi / u_s across Gamma_fs at every step.

#ifndef SEM_COUPLING_COUPLED_SIMULATION_FACADE_HPP
#define SEM_COUPLING_COUPLED_SIMULATION_FACADE_HPP

#include <mfem.hpp>
#include <memory>
#include <string>

#include "config/ConfigTypes.hpp"
#include "config/YamlConfig.hpp"
#include "coupling/FluidSolidInterface.hpp"
#include "io/WavefieldWriter.hpp"
#include "material/MaterialBase.hpp"
#include "util/ProgressLogger.hpp"
#include "operator/Operator.hpp"
#include "simulation/SimulationComponents.hpp"
#include "simulation/SimulationFacade.hpp"   // for SEM::TimingInfo
#include "simulation/SimulationRunner.hpp"
#include "srcrecv/Receiver.hpp"
#include "srcrecv/Source.hpp"
#include "timeinteg/TimeIntegrator.hpp"

namespace SEM {

using mfem::ParMesh;
using mfem::ParSubMesh;

/**
 * @class CoupledSimulationFacade
 * @tparam Dim Spatial dimension (2 or 3)
 *
 * Loads the parent ParMesh and `material.type: coupled` config, splits
 * the mesh into fluid/solid ParSubMeshes via ParSubMesh::CreateFromDomain,
 * builds FE spaces, materials, operators and time integrators per side,
 * constructs a FluidSolidInterface over the auto-detected interface bdr
 * attribute, and drives the staggered acoustic→elastic coupled time loop.
 */
template<int Dim>
class CoupledSimulationFacade {
public:
    explicit CoupledSimulationFacade(MPI_Comm comm);

    ~CoupledSimulationFacade();

    CoupledSimulationFacade(const CoupledSimulationFacade&)            = delete;
    CoupledSimulationFacade& operator=(const CoupledSimulationFacade&) = delete;
    CoupledSimulationFacade(CoupledSimulationFacade&&)                 = default;
    CoupledSimulationFacade& operator=(CoupledSimulationFacade&&)      = default;

    // -------------------------------------------------------------------------
    // Configuration / setup (fluent)
    // -------------------------------------------------------------------------

    CoupledSimulationFacade& LoadConfig(const std::string& yaml_file);
    CoupledSimulationFacade& LoadConfig(std::unique_ptr<YamlConfig> config);

    /// Load parent mesh, build submeshes, create FE spaces, materials,
    /// operators, integrators, sources, receivers and the interface.
    CoupledSimulationFacade& SetupFromConfig();

    // -------------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------------

    /// Zero the state vectors on both submeshes and arm both integrators.
    void Initialize();

    /// Enable device (GPU) memory + sync for both submeshes: state vectors,
    /// wave operators (integrators/geometry/M⁻¹/taper/rhs), sources and
    /// receiver recording buffers. Call AFTER `SetupFromConfig` and BEFORE
    /// `Run`. The fluid-solid interface already flips its own internal
    /// caches to `UseDevice(true)` during Setup so nothing extra is needed
    /// for the coupling path. No-op if built without a GPU backend.
    void DeviceInit();

    /// Advance both sub-simulations by exactly one time step. Returns true
    /// while more steps remain.
    bool Step();

    /// Initialize + drive Step() until both runners report finished.
    void Run();

    /// Reset both sub-simulations to their initial state (u=v=a=0, step=0,
    /// time=T0, attenuation memory variables zeroed, receiver buffers
    /// zeroed). Mirrors `SimulationFacade::Reset` — required for sequential
    /// multi-shot mode and coupled-domain FWI iterations.
    void Reset();

    /// Write both receiver seismograms to the output directory configured in
    /// YAML. Filenames are suffixed with `_fluid` / `_solid` so the two sides
    /// do not clobber each other.
    void SaveReceivers();

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    const YamlConfig& Config() const { return *config_; }

    ParMesh& ParentMesh() { return *parent_mesh_; }
    const ParMesh& ParentMesh() const { return *parent_mesh_; }

    SimulationComponents<Dim>& FluidComponents()             { return fluid_comp_; }
    const SimulationComponents<Dim>& FluidComponents() const { return fluid_comp_; }
    SimulationComponents<Dim>& SolidComponents()             { return solid_comp_; }
    const SimulationComponents<Dim>& SolidComponents() const { return solid_comp_; }

    MaterialBase& FluidMaterial()             { return *fluid_material_; }
    const MaterialBase& FluidMaterial() const { return *fluid_material_; }
    MaterialBase& SolidMaterial()             { return *solid_material_; }
    const MaterialBase& SolidMaterial() const { return *solid_material_; }

    WaveOperator& FluidOperator()             { return *fluid_op_; }
    const WaveOperator& FluidOperator() const { return *fluid_op_; }
    WaveOperator& SolidOperator()             { return *solid_op_; }
    const WaveOperator& SolidOperator() const { return *solid_op_; }

    SimulationRunner<Dim>& FluidRunner()             { return fluid_runner_; }
    const SimulationRunner<Dim>& FluidRunner() const { return fluid_runner_; }
    SimulationRunner<Dim>& SolidRunner()             { return solid_runner_; }
    const SimulationRunner<Dim>& SolidRunner() const { return solid_runner_; }

    PointSourceCollection* FluidSources()             { return fluid_sources_.get(); }
    PointSourceCollection* SolidSources()             { return solid_sources_.get(); }
    ReceiverArray*         FluidReceivers()           { return fluid_receivers_.get(); }
    ReceiverArray*         SolidReceivers()           { return solid_receivers_.get(); }

    int NumFluidSources()   const { return fluid_sources_ ? fluid_sources_->NumSources() : 0; }
    int NumSolidSources()   const { return solid_sources_ ? solid_sources_->NumSources() : 0; }
    int NumFluidReceivers() const { return fluid_receivers_ ? fluid_receivers_->NumReceivers() : 0; }
    int NumSolidReceivers() const { return solid_receivers_ ? solid_receivers_->NumReceivers() : 0; }

    FluidSolidInterface<Dim>&       Interface()       { return iface_; }
    const FluidSolidInterface<Dim>& Interface() const { return iface_; }

    /// Enable / disable the fluid-solid coupling terms. Default: enabled
    /// after SetupFromConfig. When disabled, fluid and solid advance
    /// independently — used by the no-coupling regression test.
    void SetCouplingEnabled(bool on) { coupling_enabled_ = on; }
    bool IsCouplingEnabled() const   { return coupling_enabled_; }

    int FluidAttribute() const { return coupled_cfg_.fluid_attribute; }
    int SolidAttribute() const { return coupled_cfg_.solid_attribute; }

    real_t Dt() const      { return dt_; }
    int    NumSteps() const { return nt_; }

    MPI_Comm Comm() const { return comm_; }
    int      Rank() const { return rank_; }
    bool     IsRoot() const { return rank_ == 0; }

    // -------------------------------------------------------------------------
    // Observability: progress / log.txt / wavefield output / CFL / summary
    //
    // These mirror the ForwardSimulation accessors so that
    // `test_forward_simulation` can wire them up uniformly regardless of
    // whether it dispatches to pure or coupled execution.
    // -------------------------------------------------------------------------

    /// Convenience accessors forwarded from the loaded YAML config.
    int         LogInterval()        const;
    std::string OutputDir()          const;
    bool        IsWavefieldOutputEnabled() const;
    int         WavefieldInterval()  const;
    std::string SummaryFile()        const;
    real_t      T0()                 const;

    /// Enable per-step progress line on stdout (and appended to log.txt under
    /// `OutputDir()`) at the given step interval. Pass 0 to disable.
    CoupledSimulationFacade& EnableProgressOutput(int interval);

    /// Attach wavefield writers for each submesh. `Run()` will call their
    /// `Init / ShouldWrite / Write / Finalize` hooks with the respective
    /// fluid / solid state vectors. Files are prefixed with "fluid_" and
    /// "solid_" so the two sides don't clobber each other.
    CoupledSimulationFacade& SetFluidWavefieldWriters(
        std::vector<std::unique_ptr<WavefieldWriter>> writers);
    CoupledSimulationFacade& SetSolidWavefieldWriters(
        std::vector<std::unique_ptr<WavefieldWriter>> writers);

    /// Run `SimulationFacade::CheckCFL` semantics on BOTH sub-operators and
    /// return true iff both pass. Call after `SetupFromConfig`. When
    /// `abort_on_violation=true`, aborts on the first violation found.
    bool CheckCFL(real_t cfl_factor, bool abort_on_violation = true);

    /// Run `SimulationFacade::CheckWavelengthSampling` semantics on BOTH
    /// submeshes and return true iff both have sufficient PPW coverage.
    /// Fluid submesh uses Vp_min (water), solid submesh uses Vs_min
    /// (shear wavelength is typically the binding constraint).
    bool CheckWavelengthSampling(real_t f_max, real_t ppw_required = 5.0,
                                 bool warn_on_insufficient = true);

    /// Write a combined summary (geometry, sources, receivers, timing) to
    /// the given path. Mirrors `ForwardSimulation::WriteSummaryToFile`.
    void WriteSummaryToFile(const std::string& path, const TimingInfo& timing);

private:
    MPI_Comm comm_;
    int rank_ = 0;
    int num_procs_ = 1;

    std::unique_ptr<YamlConfig> config_;
    CoupledMaterialConfig coupled_cfg_;

    std::unique_ptr<ParMesh> parent_mesh_;

    // Components: own the submeshes (via SetMesh) and their FE spaces / state.
    SimulationComponents<Dim> fluid_comp_;
    SimulationComponents<Dim> solid_comp_;

    std::unique_ptr<MaterialBase> fluid_material_;
    std::unique_ptr<MaterialBase> solid_material_;

    std::unique_ptr<WaveOperator>            fluid_op_;
    std::unique_ptr<WaveOperator>            solid_op_;
    std::unique_ptr<ExplicitTimeIntegrator>  fluid_integrator_;
    std::unique_ptr<ExplicitTimeIntegrator>  solid_integrator_;

    std::unique_ptr<PointSourceCollection>   fluid_sources_;
    std::unique_ptr<PointSourceCollection>   solid_sources_;
    std::unique_ptr<ReceiverArray>           fluid_receivers_;
    std::unique_ptr<ReceiverArray>           solid_receivers_;

    FluidSolidInterface<Dim> iface_;
    bool                     coupling_enabled_ = true;

    SimulationRunner<Dim> fluid_runner_;
    SimulationRunner<Dim> solid_runner_;

    real_t dt_ = 0.0;
    int    nt_ = 0;

    bool setup_done_      = false;
    bool initialized_once_ = false;

    // Progress / log.txt / wavefield state
    ProgressLogger fluid_progress_;
    ProgressLogger solid_progress_;
    std::string    log_file_;          ///< <OutputDir>/log.txt once enabled
    std::vector<std::unique_ptr<WavefieldWriter>> fluid_wavefield_writers_;
    std::vector<std::unique_ptr<WavefieldWriter>> solid_wavefield_writers_;

    // Internal setup helpers (called from SetupFromConfig)
    void SetupSourcesFromConfig();
    void SetupReceiversFromConfig();
};

using CoupledSimulationFacade2D = CoupledSimulationFacade<2>;
using CoupledSimulationFacade3D = CoupledSimulationFacade<3>;

}  // namespace SEM

#endif  // SEM_COUPLING_COUPLED_SIMULATION_FACADE_HPP
