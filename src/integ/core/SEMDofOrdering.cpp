/**
 * @file SEMDofOrdering.cpp
 * @brief Implementation of DOF ordering computation for SEM integrators
 */

#include "integ/core/SEMDofOrdering.hpp"

namespace SEM {

// =============================================================================
// SEMDofOrdering2D Implementation
// =============================================================================

void SEMDofOrdering2D::ComputeScalar(const FiniteElementSpace& fes, int order)
{
    Mesh* mesh = fes.GetMesh();
    ne = mesh->GetNE();
    ngll = order + 1;
    field_type = FieldType::Scalar;

    const int ngll2 = ngll * ngll;
    gather_map.SetSize(ne * ngll2);

    // Get lexicographic ordering (H1_QuadrilateralElement is a NodalFiniteElement)
    H1_QuadrilateralElement quad_fe(order);
    const Array<int>& lex_ref = quad_fe.GetLexicographicOrdering();

    Array<int> dof_ids;
    for (int e = 0; e < ne; e++) {
        fes.GetElementDofs(e, dof_ids);
        for (int iy = 0; iy < ngll; iy++) {
            for (int ix = 0; ix < ngll; ix++) {
                // Legacy indexing: (e * ngll + iy) * ngll + ix
                int local_idx = (e * ngll + iy) * ngll + ix;
                // Legacy lex uses column-major: lex[ix + iy * ngll]
                int lex_local = ix + iy * ngll;
                int lex_idx = lex_ref[lex_local];
                gather_map[local_idx] = dof_ids[lex_idx];
            }
        }
    }

    // Enable device memory AFTER all data is written
    // This ensures the first Read() call will sync host->device
    gather_map.GetMemory().UseDevice(true);

    // Force host→device sync for GPU builds
    SyncToDevice();
}

void SEMDofOrdering2D::ComputeVector(const FiniteElementSpace& fes, int order)
{
    Mesh* mesh = fes.GetMesh();
    ne = mesh->GetNE();
    ngll = order + 1;
    field_type = FieldType::Vector;

    const int ngll2 = ngll * ngll;
    gather_map_x.SetSize(ne * ngll2);
    gather_map_y.SetSize(ne * ngll2);

    // Get lexicographic ordering (H1_QuadrilateralElement is a NodalFiniteElement)
    H1_QuadrilateralElement quad_fe(order);
    const Array<int>& lex_ref = quad_fe.GetLexicographicOrdering();

    int dof = quad_fe.GetDof();

    Array<int> dof_ids;
    for (int e = 0; e < ne; e++) {
        fes.GetElementVDofs(e, dof_ids);

        for (int iy = 0; iy < ngll; iy++) {
            for (int ix = 0; ix < ngll; ix++) {
                int local_idx = (e * ngll + iy) * ngll + ix;
                int lex_local = ix + iy * ngll;
                int lex_x_idx = lex_ref[lex_local];
                int lex_y_idx = lex_x_idx + dof;  // y-component offset by dof

                gather_map_x[local_idx] = dof_ids[lex_x_idx];
                gather_map_y[local_idx] = dof_ids[lex_y_idx];
            }
        }
    }

    // Enable device memory AFTER all data is written
    // This ensures the first Read() call will sync host->device
    gather_map_x.GetMemory().UseDevice(true);
    gather_map_y.GetMemory().UseDevice(true);

    // Force host→device sync for GPU builds
    SyncToDevice();
}


// =============================================================================
// SEMDofOrdering3D Implementation
// =============================================================================

void SEMDofOrdering3D::ComputeScalar(const FiniteElementSpace& fes, int order)
{
    Mesh* mesh = fes.GetMesh();
    ne = mesh->GetNE();
    ngll = order + 1;
    field_type = FieldType::Scalar;

    const int ngll3 = ngll * ngll * ngll;
    gather_map.SetSize(ne * ngll3);

    // Get lexicographic ordering (H1_HexahedronElement is a NodalFiniteElement)
    H1_HexahedronElement hex_fe(order);
    const Array<int>& lex_ref = hex_fe.GetLexicographicOrdering();

    Array<int> dof_ids;
    for (int e = 0; e < ne; e++) {
        fes.GetElementDofs(e, dof_ids);
        for (int iz = 0; iz < ngll; iz++) {
            for (int iy = 0; iy < ngll; iy++) {
                for (int ix = 0; ix < ngll; ix++) {
                    int local_idx = ((e * ngll + iz) * ngll + iy) * ngll + ix;
                    int lex_local = (iz * ngll + iy) * ngll + ix;
                    int lex_idx = lex_ref[lex_local];
                    gather_map[local_idx] = dof_ids[lex_idx];
                }
            }
        }
    }

    // Enable device memory AFTER all data is written
    // This ensures the first Read() call will sync host->device
    gather_map.GetMemory().UseDevice(true);

    // Force host→device sync for GPU builds
    SyncToDevice();
}

void SEMDofOrdering3D::ComputeVector(const FiniteElementSpace& fes, int order)
{
    Mesh* mesh = fes.GetMesh();
    ne = mesh->GetNE();
    ngll = order + 1;
    field_type = FieldType::Vector;

    const int ngll3 = ngll * ngll * ngll;
    gather_map_x.SetSize(ne * ngll3);
    gather_map_y.SetSize(ne * ngll3);
    gather_map_z.SetSize(ne * ngll3);

    // Get lexicographic ordering (H1_HexahedronElement is a NodalFiniteElement)
    H1_HexahedronElement hex_fe(order);
    const Array<int>& lex_ref = hex_fe.GetLexicographicOrdering();

    int dof = hex_fe.GetDof();

    Array<int> dof_ids;
    for (int e = 0; e < ne; e++) {
        fes.GetElementVDofs(e, dof_ids);

        for (int iz = 0; iz < ngll; iz++) {
            for (int iy = 0; iy < ngll; iy++) {
                for (int ix = 0; ix < ngll; ix++) {
                    int local_idx = ((e * ngll + iz) * ngll + iy) * ngll + ix;
                    int lex_local = (iz * ngll + iy) * ngll + ix;
                    int lex_x_idx = lex_ref[lex_local];
                    int lex_y_idx = lex_x_idx + dof;
                    int lex_z_idx = lex_x_idx + 2 * dof;

                    gather_map_x[local_idx] = dof_ids[lex_x_idx];
                    gather_map_y[local_idx] = dof_ids[lex_y_idx];
                    gather_map_z[local_idx] = dof_ids[lex_z_idx];
                }
            }
        }
    }

    // Enable device memory AFTER all data is written
    // This ensures the first Read() call will sync host->device
    gather_map_x.GetMemory().UseDevice(true);
    gather_map_y.GetMemory().UseDevice(true);
    gather_map_z.GetMemory().UseDevice(true);

    // Force host→device sync for GPU builds
    SyncToDevice();
}

}  // namespace SEM
