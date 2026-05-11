/**
 * @file coupling_probe_facade.cpp
 * @brief Phase 1 probe: exercise CoupledSimulationFacade::SetupFromConfig.
 *
 * Loads a coupled-material YAML, runs the facade through mesh + submeshes +
 * materials, and emits KEY=VALUE lines describing the resulting topology so
 * the pytest driver can assert on them. Fails hard (non-zero exit) if any
 * verification inside the facade aborts.
 */

#include "coupling/CoupledSimulationFacade.hpp"
#include "common/Types.hpp"

#include <mfem.hpp>
#include <iostream>
#include <string>

using namespace mfem;
using namespace SEM;

template <int Dim>
static int RunProbe(const std::string& yaml_file, bool run_time_loop,
                    bool enable_coupling)
{
    CoupledSimulationFacade<Dim> sim(MPI_COMM_WORLD);
    sim.LoadConfig(yaml_file).SetupFromConfig();
    sim.SetCouplingEnabled(enable_coupling);

    const ParMesh& parent = sim.ParentMesh();
    const ParMesh& fluid  = sim.FluidComponents().Mesh();
    const ParMesh& solid  = sim.SolidComponents().Mesh();

    long long p_ne = parent.GetNE(), f_ne = fluid.GetNE(), s_ne = solid.GetNE();
    long long p_tot = 0, f_tot = 0, s_tot = 0;
    MPI_Reduce(&p_ne, &p_tot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&f_ne, &f_tot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&s_ne, &s_tot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    const int f_vdim = sim.FluidComponents().FES().GetVDim();
    const int s_vdim = sim.SolidComponents().FES().GetVDim();
    const HYPRE_BigInt f_dofs = sim.FluidComponents().GlobalNumDOFs();
    const HYPRE_BigInt s_dofs = sim.SolidComponents().GlobalNumDOFs();

    const auto fmat = sim.FluidMaterial().GetType();
    const auto smat = sim.SolidMaterial().GetType();

    if (Mpi::Root()) {
        std::cout << "DIM=" << Dim << "\n";
        std::cout << "PARENT_NE=" << p_tot << "\n";
        std::cout << "FLUID_ATTR=" << sim.FluidAttribute() << "\n";
        std::cout << "SOLID_ATTR=" << sim.SolidAttribute() << "\n";
        std::cout << "FLUID_NE=" << f_tot << "\n";
        std::cout << "SOLID_NE=" << s_tot << "\n";
        std::cout << "FLUID_VDIM=" << f_vdim << "\n";
        std::cout << "SOLID_VDIM=" << s_vdim << "\n";
        std::cout << "FLUID_GLOBAL_DOFS=" << f_dofs << "\n";
        std::cout << "SOLID_GLOBAL_DOFS=" << s_dofs << "\n";
        std::cout << "FLUID_MATERIAL=" << MaterialTypeToString(fmat) << "\n";
        std::cout << "SOLID_MATERIAL=" << MaterialTypeToString(smat) << "\n";
        std::cout << "NUM_FLUID_SOURCES="   << sim.NumFluidSources()   << "\n";
        std::cout << "NUM_SOLID_SOURCES="   << sim.NumSolidSources()   << "\n";
        std::cout << "NUM_FLUID_RECEIVERS=" << sim.NumFluidReceivers() << "\n";
        std::cout << "NUM_SOLID_RECEIVERS=" << sim.NumSolidReceivers() << "\n";
        std::cout << "DT=" << sim.Dt() << "\n";
        std::cout << "NT=" << sim.NumSteps() << "\n";
    }

    if (run_time_loop) {
        sim.Run();
        sim.SaveReceivers();
        if (Mpi::Root()) std::cout << "TIME_LOOP=completed\n";
    }

    if (Mpi::Root()) std::cout << "PROBE_OK=1\n";
    return 0;
}

int main(int argc, char* argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string yaml_file;
    int dim = 3;
    bool run_loop = false;
    bool enable_coupling = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--config" && i + 1 < argc) yaml_file = argv[++i];
        else if (a == "--dim"    && i + 1 < argc) dim       = std::stoi(argv[++i]);
        else if (a == "--run")                    run_loop  = true;
        else if (a == "--no-coupling")            enable_coupling = false;
    }
    if (yaml_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: coupling_probe_facade --config FILE.yaml "
                         "[--dim 2|3] [--run]\n";
        }
        return 1;
    }

    if (dim == 2) return RunProbe<2>(yaml_file, run_loop, enable_coupling);
    if (dim == 3) return RunProbe<3>(yaml_file, run_loop, enable_coupling);
    if (Mpi::Root()) std::cerr << "Unsupported --dim " << dim << "\n";
    return 1;
}
