// Probe: verify GMSH Physical Volume -> MFEM attribute inheritance and
// ParSubMesh::CreateFromDomain behaviour. Emits KEY=VALUE lines so the
// pytest driver can assert on them without parsing free-form output.
//
// Usage:
//   mpirun -np N coupling_probe_gmsh --mesh test.msh \
//                                    [--fluid-attr 1] [--solid-attr 2]

#include <mfem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>

using namespace mfem;

static void PrintAttrHistogram(const char* label, const ParMesh& pmesh) {
    std::map<int, long long> local;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        local[pmesh.GetAttribute(e)]++;
    }
    // Collect a global set of attributes first (so every rank iterates the same keys).
    std::set<int> keys_local;
    for (auto& kv : local) keys_local.insert(kv.first);
    int local_nkeys = (int)keys_local.size();
    int max_nkeys = 0;
    MPI_Allreduce(&local_nkeys, &max_nkeys, 1, MPI_INT, MPI_MAX, pmesh.GetComm());
    // Reduce via union of sorted key arrays across ranks.
    // Simple approach: gather min/max attribute and sum counts per attribute id
    // across [min, max]. This is O((max-min)*nranks) but fine for a probe.
    int local_min = local.empty() ? std::numeric_limits<int>::max()
                                  : local.begin()->first;
    int local_max = local.empty() ? std::numeric_limits<int>::min()
                                  : local.rbegin()->first;
    int global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, pmesh.GetComm());
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, pmesh.GetComm());

    for (int a = global_min; a <= global_max; ++a) {
        long long mine = 0;
        auto it = local.find(a);
        if (it != local.end()) mine = it->second;
        long long total = 0;
        MPI_Reduce(&mine, &total, 1, MPI_LONG_LONG, MPI_SUM, 0, pmesh.GetComm());
        if (Mpi::Root() && total > 0) {
            std::cout << label << "_ATTR_" << a << "=" << total << "\n";
        }
    }
}

static void PrintBdrAttrSet(const char* label, const ParMesh& pmesh) {
    std::set<int> local;
    for (int b = 0; b < pmesh.GetNBE(); ++b) {
        local.insert(pmesh.GetBdrAttribute(b));
    }
    int local_min = local.empty() ? std::numeric_limits<int>::max()
                                  : *local.begin();
    int local_max = local.empty() ? std::numeric_limits<int>::min()
                                  : *local.rbegin();
    int gmin, gmax;
    MPI_Allreduce(&local_min, &gmin, 1, MPI_INT, MPI_MIN, pmesh.GetComm());
    MPI_Allreduce(&local_max, &gmax, 1, MPI_INT, MPI_MAX, pmesh.GetComm());

    std::string acc;
    for (int a = gmin; a <= gmax; ++a) {
        int mine = local.count(a) ? 1 : 0;
        int any = 0;
        MPI_Reduce(&mine, &any, 1, MPI_INT, MPI_LOR, 0, pmesh.GetComm());
        if (Mpi::Root() && any) {
            if (!acc.empty()) acc += ",";
            acc += std::to_string(a);
        }
    }
    if (Mpi::Root()) {
        std::cout << label << "_BDR_ATTRS=" << acc << "\n";
    }
}

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string mesh_file;
    std::string dump_dir;
    int fluid_attr = 1;
    int solid_attr = 2;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mesh" && i + 1 < argc) mesh_file = argv[++i];
        else if (a == "--dump-dir" && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--fluid-attr" && i + 1 < argc) fluid_attr = std::stoi(argv[++i]);
        else if (a == "--solid-attr" && i + 1 < argc) solid_attr = std::stoi(argv[++i]);
    }

    if (mesh_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: coupling_probe_gmsh --mesh FILE.msh "
                         "[--fluid-attr 1] [--solid-attr 2]\n";
        }
        return 1;
    }

    // Load serial mesh (rank 0 path is fine for a probe; ParMesh will redistribute).
    Mesh smesh(mesh_file.c_str(), 1, 1);
    if (Mpi::Root()) {
        std::cout << "PARENT_DIM=" << smesh.Dimension() << "\n";
        std::cout << "PARENT_NE_SERIAL=" << smesh.GetNE() << "\n";
    }

    ParMesh pmesh(MPI_COMM_WORLD, smesh);
    long long local_ne = pmesh.GetNE();
    long long total_ne = 0;
    MPI_Reduce(&local_ne, &total_ne, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (Mpi::Root()) std::cout << "PARENT_NE_PARALLEL=" << total_ne << "\n";

    // Per-rank breakdown: how many fluid/solid elements each rank owns.
    long long local_f = 0, local_s = 0;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        int a = pmesh.GetAttribute(e);
        if (a == fluid_attr) ++local_f;
        else if (a == solid_attr) ++local_s;
    }
    int nranks = Mpi::WorldSize();
    std::vector<long long> all_f(nranks, 0), all_s(nranks, 0), all_t(nranks, 0);
    MPI_Gather(&local_f, 1, MPI_LONG_LONG, all_f.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_s, 1, MPI_LONG_LONG, all_s.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_ne, 1, MPI_LONG_LONG, all_t.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (Mpi::Root()) {
        for (int r = 0; r < nranks; ++r) {
            std::cout << "PARENT_RANK_" << r << "=total:" << all_t[r]
                      << ",fluid:" << all_f[r] << ",solid:" << all_s[r] << "\n";
        }
    }

    PrintAttrHistogram("PARENT", pmesh);
    PrintBdrAttrSet("PARENT", pmesh);

    // Create submeshes.
    Array<int> fa(1); fa[0] = fluid_attr;
    Array<int> sa(1); sa[0] = solid_attr;

    ParSubMesh fluid_sub = ParSubMesh::CreateFromDomain(pmesh, fa);
    ParSubMesh solid_sub = ParSubMesh::CreateFromDomain(pmesh, sa);

    long long f_ne = fluid_sub.GetNE(), s_ne = solid_sub.GetNE();
    long long f_tot = 0, s_tot = 0;
    MPI_Reduce(&f_ne, &f_tot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&s_ne, &s_tot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (Mpi::Root()) {
        std::cout << "FLUID_SUB_NE=" << f_tot << "\n";
        std::cout << "SOLID_SUB_NE=" << s_tot << "\n";
    }

    PrintBdrAttrSet("FLUID_SUB", fluid_sub);
    PrintBdrAttrSet("SOLID_SUB", solid_sub);

    if (!dump_dir.empty()) {
        if (Mpi::Root()) {
            mkdir(dump_dir.c_str(), 0755);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        auto dump = [&](const char* prefix, ParMesh& m) {
            std::ostringstream fname;
            fname << dump_dir << "/" << prefix << "."
                  << std::setw(6) << std::setfill('0') << Mpi::WorldRank();
            std::ofstream ofs(fname.str());
            ofs.precision(16);
            m.ParPrint(ofs);
        };
        dump("parent", pmesh);
        dump("fluid",  fluid_sub);
        dump("solid",  solid_sub);

        if (Mpi::Root()) {
            std::cout << "DUMP_DIR=" << dump_dir << "\n"
                      << "GLVIS_HINT=glvis -np " << Mpi::WorldSize()
                      << " -m " << dump_dir << "/parent  (or fluid/solid)\n";
        }
    }

    if (Mpi::Root()) std::cout << "PROBE_OK=1\n";
    return 0;
}
