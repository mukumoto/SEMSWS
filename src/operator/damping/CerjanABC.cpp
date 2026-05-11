/**
 * @file CerjanABC.cpp
 * @brief Implementation of Cerjan sponge absorbing boundary condition
 */

#include "operator/damping/CerjanABC.hpp"
#include "operator/Operator.hpp"  // For DampingConfig

#include <algorithm>
#include <cmath>

namespace SEM {
namespace damping {

void ComputeCerjanTaper2D(
    ParFiniteElementSpace& fes,
    const DampingConfig& config,
    int ngll,
    Vector& taper)
{
    const int ne = fes.GetMesh()->GetNE();
    const int64_t total = (int64_t)ne * ngll * ngll;
    MFEM_VERIFY(total <= INT_MAX,
                "ComputeCerjanTaper2D size (" << total << ") exceeds int32 limit.");

    taper.SetSize(total);
    taper = 1.0;  // Interior: no damping

    // Get global bounding box
    Vector bbmin, bbmax;
    fes.GetParMesh()->GetBoundingBox(bbmin, bbmax);

    Vector bbmin_all(bbmin.Size()), bbmax_all(bbmax.Size());
    MPI_Allreduce(bbmin.GetData(), bbmin_all.GetData(), bbmin.Size(),
                  MPITypeMap<real_t>::mpi_type, MPI_MIN, fes.GetComm());
    MPI_Allreduce(bbmax.GetData(), bbmax_all.GetData(), bbmax.Size(),
                  MPITypeMap<real_t>::mpi_type, MPI_MAX, fes.GetComm());

    Vector p0(2);
    p0(0) = bbmin_all(0);
    p0(1) = bbmin_all(1);
    Vector bb_rel_max = bbmax_all;
    bb_rel_max -= p0;

    // Determine which boundary sides are active
    const int nattr = static_cast<int>(config.attrs.size());
    bool has_bottom = false, has_right = false, has_top = false, has_left = false;
    for (int ia = 0; ia < nattr; ia++) {
        if (config.attrs[ia] == 1) has_bottom = true;
        if (config.attrs[ia] == 2) has_right = true;
        if (config.attrs[ia] == 3) has_top = true;
        if (config.attrs[ia] == 4) has_left = true;
    }

    // Cerjan coefficient: a = sqrt(-log(1 - abpc/100))
    // alpha[0] is abpc (percentage), e.g., 6 means 6% decay at boundary
    real_t abpc_x = config.alpha[0];
    real_t abpc_y = config.alpha[1];
    real_t a_x = std::sqrt(-std::log(1.0 - abpc_x / 100.0));
    real_t a_y = std::sqrt(-std::log(1.0 - abpc_y / 100.0));

    for (int ie = 0; ie < ne; ie++)
    {
        Vector cent(2);
        fes.GetElementTransformation(ie)->Transform(
            Geometries.GetCenter(fes.GetParMesh()->GetElementBaseGeometry(ie)), cent);
        cent -= p0;

        // Check if element is near any sponge boundary
        bool near_bottom = has_bottom && (cent(1) <= config.abc_lengths[1]);
        bool near_right  = has_right  && (cent(0) >= (bb_rel_max(0) - config.abc_lengths[0]));
        bool near_top    = has_top    && (cent(1) >= (bb_rel_max(1) - config.abc_lengths[1]));
        bool near_left   = has_left   && (cent(0) <= config.abc_lengths[0]);

        if (!near_bottom && !near_right && !near_top && !near_left)
            continue;

        const FiniteElement* fe = fes.GetFE(ie);
        const IntegrationRule& pdof = fe->GetNodes();

        for (int j = 0; j < pdof.GetNPoints(); j++)
        {
            Vector geo;
            fes.GetElementTransformation(ie)->Transform(pdof.IntPoint(j), geo);
            geo -= p0;

            // Compute normalized distance from interior edge for each direction
            // d = 0 at interior edge, d = 1 at domain boundary
            real_t dx = 0.0, dy = 0.0;

            if (near_left) {
                real_t d = (config.abc_lengths[0] - geo(0)) / config.abc_lengths[0];
                dx = std::max(dx, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }
            if (near_right) {
                real_t d = (config.abc_lengths[0] - (bb_rel_max(0) - geo(0))) / config.abc_lengths[0];
                dx = std::max(dx, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }
            if (near_bottom) {
                real_t d = (config.abc_lengths[1] - geo(1)) / config.abc_lengths[1];
                dy = std::max(dy, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }
            if (near_top) {
                real_t d = (config.abc_lengths[1] - (bb_rel_max(1) - geo(1))) / config.abc_lengths[1];
                dy = std::max(dy, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }

            // Cerjan taper: min of per-direction tapers (strongest damping at corners)
            real_t taper_x = (dx > 0.0) ? std::exp(-a_x * a_x * dx * dx) : 1.0;
            real_t taper_y = (dy > 0.0) ? std::exp(-a_y * a_y * dy * dy) : 1.0;
            real_t taper_val = std::min(taper_x, taper_y);

            int idx = ie * ngll * ngll + j;
            taper(idx) = std::min(taper(idx), taper_val);
        }
    }
}


void ComputeCerjanTaper3D(
    ParFiniteElementSpace& fes,
    const DampingConfig& config,
    int ngll,
    Vector& taper)
{
    const int ne = fes.GetMesh()->GetNE();
    const int ngll3 = ngll * ngll * ngll;
    const int64_t total = (int64_t)ne * ngll3;
    MFEM_VERIFY(total <= INT_MAX,
                "ComputeCerjanTaper3D size (" << total << ") exceeds int32 limit.");

    taper.SetSize(total);
    taper = 1.0;

    // Get global bounding box
    Vector bbmin, bbmax;
    fes.GetParMesh()->GetBoundingBox(bbmin, bbmax);

    Vector bbmin_all(bbmin.Size()), bbmax_all(bbmax.Size());
    MPI_Allreduce(bbmin.GetData(), bbmin_all.GetData(), bbmin.Size(),
                  MPITypeMap<real_t>::mpi_type, MPI_MIN, fes.GetComm());
    MPI_Allreduce(bbmax.GetData(), bbmax_all.GetData(), bbmax.Size(),
                  MPITypeMap<real_t>::mpi_type, MPI_MAX, fes.GetComm());

    Vector p0(3);
    p0(0) = bbmin_all(0);
    p0(1) = bbmin_all(1);
    p0(2) = bbmin_all(2);
    Vector bb_rel_max = bbmax_all;
    bb_rel_max -= p0;

    // 3D attrs: 1=Bottom(Z-), 2=Front(Y-), 3=Right(X+), 4=Back(Y+), 5=Left(X-), 6=Top(Z+)
    const int nattr = static_cast<int>(config.attrs.size());
    bool has_left = false, has_right = false;
    bool has_front = false, has_back = false;
    bool has_bottom = false, has_top = false;
    for (int ia = 0; ia < nattr; ia++) {
        if (config.attrs[ia] == 1) has_bottom = true;
        if (config.attrs[ia] == 2) has_front = true;
        if (config.attrs[ia] == 3) has_right = true;
        if (config.attrs[ia] == 4) has_back = true;
        if (config.attrs[ia] == 5) has_left = true;
        if (config.attrs[ia] == 6) has_top = true;
    }

    real_t abpc_x = config.alpha[0];
    real_t abpc_y = config.alpha[1];
    real_t abpc_z = config.alpha[2];
    real_t a_x = std::sqrt(-std::log(1.0 - abpc_x / 100.0));
    real_t a_y = std::sqrt(-std::log(1.0 - abpc_y / 100.0));
    real_t a_z = std::sqrt(-std::log(1.0 - abpc_z / 100.0));

    for (int ie = 0; ie < ne; ie++)
    {
        Vector cent(3);
        fes.GetElementTransformation(ie)->Transform(
            Geometries.GetCenter(fes.GetParMesh()->GetElementBaseGeometry(ie)), cent);
        cent -= p0;

        bool near_left   = has_left   && (cent(0) <= config.abc_lengths[0]);
        bool near_right  = has_right  && (cent(0) >= (bb_rel_max(0) - config.abc_lengths[0]));
        bool near_front  = has_front  && (cent(1) <= config.abc_lengths[1]);
        bool near_back   = has_back   && (cent(1) >= (bb_rel_max(1) - config.abc_lengths[1]));
        bool near_bottom = has_bottom && (cent(2) <= config.abc_lengths[2]);
        bool near_top    = has_top    && (cent(2) >= (bb_rel_max(2) - config.abc_lengths[2]));

        if (!near_left && !near_right && !near_front && !near_back &&
            !near_bottom && !near_top)
            continue;

        const FiniteElement* fe = fes.GetFE(ie);
        const IntegrationRule& pdof = fe->GetNodes();

        for (int j = 0; j < pdof.GetNPoints(); j++)
        {
            Vector geo;
            fes.GetElementTransformation(ie)->Transform(pdof.IntPoint(j), geo);
            geo -= p0;

            real_t dx = 0.0, dy = 0.0, dz = 0.0;

            if (near_left) {
                real_t d = (config.abc_lengths[0] - geo(0)) / config.abc_lengths[0];
                dx = std::max(dx, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }
            if (near_right) {
                real_t d = (config.abc_lengths[0] - (bb_rel_max(0) - geo(0))) / config.abc_lengths[0];
                dx = std::max(dx, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }
            if (near_front) {
                real_t d = (config.abc_lengths[1] - geo(1)) / config.abc_lengths[1];
                dy = std::max(dy, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }
            if (near_back) {
                real_t d = (config.abc_lengths[1] - (bb_rel_max(1) - geo(1))) / config.abc_lengths[1];
                dy = std::max(dy, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }
            if (near_bottom) {
                real_t d = (config.abc_lengths[2] - geo(2)) / config.abc_lengths[2];
                dz = std::max(dz, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }
            if (near_top) {
                real_t d = (config.abc_lengths[2] - (bb_rel_max(2) - geo(2))) / config.abc_lengths[2];
                dz = std::max(dz, std::max(real_t(0.0), std::min(real_t(1.0), d)));
            }

            real_t taper_x = (dx > 0.0) ? std::exp(-a_x * a_x * dx * dx) : 1.0;
            real_t taper_y = (dy > 0.0) ? std::exp(-a_y * a_y * dy * dy) : 1.0;
            real_t taper_z = (dz > 0.0) ? std::exp(-a_z * a_z * dz * dz) : 1.0;
            real_t taper_val = std::min({taper_x, taper_y, taper_z});

            int idx = ie * ngll3 + j;
            taper(idx) = std::min(taper(idx), taper_val);
        }
    }
}


void PrintCerjanStats(
    const Vector& taper,
    MPI_Comm comm,
    const char* label)
{
    real_t local_min = 1.0;
    real_t local_max = 0.0;

    for (int i = 0; i < taper.Size(); i++) {
        local_min = std::min(local_min, taper(i));
        local_max = std::max(local_max, taper(i));
    }

    real_t global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1,
                  MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1,
                  MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        mfem::out << "  " << label << ":" << std::endl;
        mfem::out << "    Min taper: " << global_min
                  << " (at boundary)" << std::endl;
        mfem::out << "    Max taper: " << global_max
                  << " (interior, should be 1.0)" << std::endl;
    }
}

}  // namespace damping
}  // namespace SEM
