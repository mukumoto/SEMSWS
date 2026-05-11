/**
 * @file MaterialField.cpp
 * @brief Implementation of MaterialField and MaterialField3D storage classes
 *
 * Also contains PrintInfo() implementations for base material classes.
 */

#include "material/Material.hpp"
#include "util/FESOrder.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "material/AcousticMaterialBase.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace SEM {

// =============================================================================
// MaterialField Implementation (2D)
// =============================================================================

MaterialField::MaterialField(int ne, int ngllx, int nglly)
    : ne_(ne), ngllx_(ngllx), nglly_(nglly), stride_e_(nglly * ngllx)
{
    data_.SetSize(ne * stride_e_);
    data_ = 0.0;
}

MaterialField::MaterialField(const MaterialField& other)
    : data_(other.data_), ne_(other.ne_), ngllx_(other.ngllx_),
      nglly_(other.nglly_), stride_e_(other.stride_e_)
{
}

MaterialField::MaterialField(MaterialField&& other) noexcept
    : data_(std::move(other.data_)), ne_(other.ne_), ngllx_(other.ngllx_),
      nglly_(other.nglly_), stride_e_(other.stride_e_)
{
    other.ne_ = other.ngllx_ = other.nglly_ = other.stride_e_ = 0;
}

MaterialField& MaterialField::operator=(const MaterialField& other)
{
    if (this != &other) {
        data_ = other.data_;
        ne_ = other.ne_;
        ngllx_ = other.ngllx_;
        nglly_ = other.nglly_;
        stride_e_ = other.stride_e_;
    }
    return *this;
}

MaterialField& MaterialField::operator=(MaterialField&& other) noexcept
{
    if (this != &other) {
        data_ = std::move(other.data_);
        ne_ = other.ne_;
        ngllx_ = other.ngllx_;
        nglly_ = other.nglly_;
        stride_e_ = other.stride_e_;
        other.ne_ = other.ngllx_ = other.nglly_ = other.stride_e_ = 0;
    }
    return *this;
}

void MaterialField::ProjectCoefficient(Coefficient& coef,
                                        const ParFiniteElementSpace& fes,
                                        const IntegrationRule& ir)
{
    real_t* data = data_.HostWrite();

    for (int e = 0; e < ne_; e++)
    {
        ElementTransformation* Tr = fes.GetElementTransformation(e);

        for (int j = 0; j < nglly_; j++)
        {
            for (int i = 0; i < ngllx_; i++)
            {
                int ip_idx = j * ngllx_ + i;
                const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                Tr->SetIntPoint(&ip);

                int flat_idx = e * stride_e_ + j * ngllx_ + i;
                data[flat_idx] = coef.Eval(*Tr, ip);
            }
        }
    }
}

void MaterialField::CopyFrom(const MaterialField& other)
{
    MFEM_VERIFY(data_.Size() == other.data_.Size(),
                "MaterialField::CopyFrom size mismatch");
    data_ = other.data_;
}

void MaterialField::SetConstant(real_t value)
{
    data_.HostWrite();  // Mark host as valid, device as invalid
    data_ = value;
}

void MaterialField::Scale(real_t factor)
{
    data_ *= factor;
}

real_t MaterialField::Min() const
{
    if (data_.Size() == 0) return 0.0;
    const real_t* d = data_.GetData();
    return *std::min_element(d, d + data_.Size());
}

real_t MaterialField::Max() const
{
    if (data_.Size() == 0) return 0.0;
    const real_t* d = data_.GetData();
    return *std::max_element(d, d + data_.Size());
}

void MaterialField::PrintStats(std::ostream& os) const
{
    os << "MaterialField: " << ne_ << " elements, "
       << ngllx_ << "x" << nglly_ << " GLL points\n";
    os << "  Size: " << data_.Size() << " (" << MemoryUsage() / 1024 << " KB)\n";
    os << "  Range: [" << Min() << ", " << Max() << "]\n";
    os << "  L2 norm: " << Norml2() << "\n";
}

void MaterialField::ToParGridFunction(const ParFiniteElementSpace& fes,
                                       ParGridFunction& gf) const
{
    // Ensure grid function is set up
    if (gf.Size() != fes.GetVSize()) {
        gf.SetSpace(const_cast<ParFiniteElementSpace*>(&fes));
    }

    // Get the element DOF table
    const int ne = fes.GetNE();
    MFEM_VERIFY(ne == ne_, "Element count mismatch: fes has " << ne
                << " elements, MaterialField has " << ne_);

    // ParSubMesh partitions can leave some ranks with zero local
    // elements on the smaller domain (e.g. the solid submesh when the
    // overall mesh is split 4-ways). `ParFiniteElementSpace::GetFE(0)`
    // on such an empty rank dereferences a null element table and
    // segfaults; bail out cleanly so the other ranks still produce
    // output.
    if (ne == 0) return;

    const real_t* mat_data = data_.GetData();
    real_t* gf_data = gf.GetData();

    // Get lexicographic ordering from the finite element
    // MFEM's GetLexicographicOrdering returns dof_map where:
    //   dof_map[lex_pos] = internal_dof_idx
    // Lex position uses column-major ordering: i + j * ngllx (for 2D)
    // SEMSWS always uses H1 elements which are NodalFiniteElements
    const FiniteElement* fe = fes.GetFE(0);
    const NodalFiniteElement* nodal_fe = static_cast<const NodalFiniteElement*>(fe);
    const Array<int>& lex_ref = nodal_fe->GetLexicographicOrdering();

    Array<int> lex_ordering;
    if (lex_ref.Size() > 0) {
        lex_ordering.SetSize(lex_ref.Size());
        for (int i = 0; i < lex_ref.Size(); i++) {
            lex_ordering[i] = lex_ref[i];
        }
    } else {
        // Identity mapping: lex_pos = internal_dof_idx
        lex_ordering.SetSize(ngllx_ * nglly_);
        for (int i = 0; i < ngllx_ * nglly_; i++) {
            lex_ordering[i] = i;
        }
    }

    // For each element, map GLL points to DOFs
    Array<int> dofs;
    for (int e = 0; e < ne; e++)
    {
        fes.GetElementDofs(e, dofs);
        const int ndofs = dofs.Size();

        // Verify DOF count matches GLL points
        MFEM_VERIFY(ndofs == ngllx_ * nglly_,
                    "DOF count mismatch: element " << e << " has " << ndofs
                    << " DOFs, expected " << ngllx_ * nglly_);

        // Map material values to DOFs using lexicographic ordering
        // MaterialField stores data in row-major order: j * ngllx + i (y*nx + x)
        // MFEM lex ordering uses column-major: i + j * ngllx (x + y*nx)
        // lex_ordering[lex_pos] = internal_dof_idx
        for (int j = 0; j < nglly_; j++)
        {
            for (int i = 0; i < ngllx_; i++)
            {
                // MaterialField uses row-major: j * ngllx + i
                int mat_idx = e * stride_e_ + j * ngllx_ + i;

                // MFEM lex ordering uses column-major: i + j * ngllx
                int lex_pos = i + j * ngllx_;
                int internal_idx = lex_ordering[lex_pos];
                int dof_idx = dofs[internal_idx];

                // Handle negative DOF indices (orientation)
                if (dof_idx < 0) {
                    dof_idx = -1 - dof_idx;
                }

                gf_data[dof_idx] = mat_data[mat_idx];
            }
        }
    }
}

ParGridFunction* MaterialField::ToParGridFunction(ParFiniteElementSpace* fes) const
{
    ParGridFunction* gf = new ParGridFunction(fes);
    ToParGridFunction(*fes, *gf);
    return gf;
}

MaterialField MaterialField::FromParGridFunction(const ParFiniteElementSpace& fes,
                                                  const ParGridFunction& gf)
{
    const int ne = fes.GetNE();
    const int order = SafeFESOrder(fes);
    const int ngllx = order + 1;
    const int nglly = order + 1;

    MaterialField field(ne, ngllx, nglly);
    real_t* mat_data = field.HostWrite();
    const real_t* gf_data = gf.GetData();

    // Get lexicographic ordering (same logic as ToParGridFunction)
    const FiniteElement* fe = fes.GetFE(0);
    const NodalFiniteElement* nodal_fe = static_cast<const NodalFiniteElement*>(fe);
    const Array<int>& lex_ref = nodal_fe->GetLexicographicOrdering();

    Array<int> lex_ordering;
    if (lex_ref.Size() > 0) {
        lex_ordering.SetSize(lex_ref.Size());
        for (int i = 0; i < lex_ref.Size(); i++) {
            lex_ordering[i] = lex_ref[i];
        }
    } else {
        lex_ordering.SetSize(ngllx * nglly);
        for (int i = 0; i < ngllx * nglly; i++) {
            lex_ordering[i] = i;
        }
    }

    const int stride_e = nglly * ngllx;
    Array<int> dofs;
    for (int e = 0; e < ne; e++)
    {
        fes.GetElementDofs(e, dofs);

        for (int j = 0; j < nglly; j++)
        {
            for (int i = 0; i < ngllx; i++)
            {
                int mat_idx = e * stride_e + j * ngllx + i;
                int lex_pos = i + j * ngllx;
                int internal_idx = lex_ordering[lex_pos];
                int dof_idx = dofs[internal_idx];

                if (dof_idx < 0) {
                    dof_idx = -1 - dof_idx;
                }

                mat_data[mat_idx] = gf_data[dof_idx];
            }
        }
    }

    return field;
}


// =============================================================================
// MaterialField3D Implementation
// =============================================================================

MaterialField3D::MaterialField3D(int ne, int ngllx, int nglly, int ngllz)
    : ne_(ne), ngllx_(ngllx), nglly_(nglly), ngllz_(ngllz),
      stride_k_(nglly * ngllx), stride_e_(ngllz * nglly * ngllx)
{
    data_.SetSize(ne * stride_e_);
    data_ = 0.0;
    // Enable device (GPU) memory for use in GPU kernels
    data_.UseDevice(true);
}

void MaterialField3D::ProjectCoefficient(
    Coefficient& coef, const ParFiniteElementSpace& fes, const IntegrationRule& ir)
{
    real_t* data = data_.HostWrite();

    for (int e = 0; e < ne_; e++)
    {
        ElementTransformation* Tr = fes.GetElementTransformation(e);

        for (int k = 0; k < ngllz_; k++)
        {
            for (int j = 0; j < nglly_; j++)
            {
                for (int i = 0; i < ngllx_; i++)
                {
                    int ip_idx = k * nglly_ * ngllx_ + j * ngllx_ + i;
                    const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                    Tr->SetIntPoint(&ip);

                    int flat_idx = e * stride_e_ + k * stride_k_ + j * ngllx_ + i;
                    data[flat_idx] = coef.Eval(*Tr, ip);
                }
            }
        }
    }
}

void MaterialField3D::SetConstant(real_t value)
{
    data_.HostWrite();  // Mark host as valid, device as invalid
    data_ = value;
}

void MaterialField3D::ToParGridFunction(const ParFiniteElementSpace& fes,
                                         ParGridFunction& gf) const
{
    // Ensure grid function is set up
    if (gf.Size() != fes.GetVSize()) {
        gf.SetSpace(const_cast<ParFiniteElementSpace*>(&fes));
    }

    // Initialize to zero to avoid uninitialized memory issues
    gf = 0.0;

    // Get the element DOF table
    const int ne = fes.GetNE();
    MFEM_VERIFY(ne == ne_, "Element count mismatch: fes has " << ne
                << " elements, MaterialField3D has " << ne_);

    // Skip cleanly on ranks that hold no local elements — same
    // ParSubMesh-partitioning concern as the 2D overload above.
    if (ne == 0) return;

    const real_t* mat_data = data_.GetData();
    real_t* gf_data = gf.GetData();

    // Get lexicographic ordering from the finite element
    // MFEM's GetLexicographicOrdering returns dof_map where:
    //   dof_map[lex_pos] = internal_dof_idx
    // Lex position uses column-major ordering: i + j * ngllx + k * ngllx * nglly
    // SEMSWS always uses H1 elements which are NodalFiniteElements
    const FiniteElement* fe = fes.GetFE(0);
    const NodalFiniteElement* nodal_fe = static_cast<const NodalFiniteElement*>(fe);
    const Array<int>& lex_ref = nodal_fe->GetLexicographicOrdering();

    const int ngll3 = ngllx_ * nglly_ * ngllz_;

    Array<int> lex_ordering;
    if (lex_ref.Size() > 0) {
        MFEM_VERIFY(lex_ref.Size() == ngll3, "Lex ordering size mismatch: "
                    << lex_ref.Size() << " != " << ngll3);
        lex_ordering.SetSize(lex_ref.Size());
        for (int i = 0; i < lex_ref.Size(); i++) {
            lex_ordering[i] = lex_ref[i];
        }
    } else {
        // Identity mapping: lex_pos = internal_dof_idx
        lex_ordering.SetSize(ngll3);
        for (int i = 0; i < ngll3; i++) {
            lex_ordering[i] = i;
        }
    }

    // For each element, map GLL points to DOFs
    Array<int> dofs;
    for (int e = 0; e < ne; e++)
    {
        fes.GetElementDofs(e, dofs);
        const int ndofs = dofs.Size();

        // Verify DOF count matches GLL points
        MFEM_VERIFY(ndofs == ngll3,
                    "DOF count mismatch: element " << e << " has " << ndofs
                    << " DOFs, expected " << ngll3);

        // Map material values to DOFs using lexicographic ordering
        // MaterialField3D stores data in row-major order: k * ny * nx + j * nx + i
        // MFEM lex ordering uses column-major: i + j * nx + k * nx * ny
        // lex_ordering[lex_pos] = internal_dof_idx
        for (int k = 0; k < ngllz_; k++)
        {
            for (int j = 0; j < nglly_; j++)
            {
                for (int i = 0; i < ngllx_; i++)
                {
                    // MaterialField3D uses row-major: k * ny * nx + j * nx + i
                    int mat_idx = e * stride_e_ + k * stride_k_ + j * ngllx_ + i;

                    // MFEM lex ordering uses column-major: i + j * nx + k * nx * ny
                    int lex_pos = i + j * ngllx_ + k * ngllx_ * nglly_;
                    int internal_idx = lex_ordering[lex_pos];
                    int dof_idx = dofs[internal_idx];

                    // Handle negative DOF indices (orientation)
                    if (dof_idx < 0) {
                        dof_idx = -1 - dof_idx;
                    }

                    gf_data[dof_idx] = mat_data[mat_idx];
                }
            }
        }
    }
}

ParGridFunction* MaterialField3D::ToParGridFunction(ParFiniteElementSpace* fes) const
{
    ParGridFunction* gf = new ParGridFunction(fes);
    ToParGridFunction(*fes, *gf);
    return gf;
}

// Inverse of ToParGridFunction above — pull DOF values back into a 3D
// MaterialField using the same lexicographic mapping.
MaterialField3D MaterialField3D::FromParGridFunction(const ParFiniteElementSpace& fes,
                                                    const ParGridFunction& gf)
{
    const int ne    = fes.GetNE();
    const int order = SafeFESOrder(fes);
    const int ngllx = order + 1;
    const int nglly = order + 1;
    const int ngllz = order + 1;

    MaterialField3D field(ne, ngllx, nglly, ngllz);
    real_t* mat_data = field.HostWrite();
    const real_t* gf_data = gf.GetData();

    const FiniteElement* fe = fes.GetFE(0);
    const NodalFiniteElement* nodal_fe =
        static_cast<const NodalFiniteElement*>(fe);
    const Array<int>& lex_ref = nodal_fe->GetLexicographicOrdering();

    const int ngll3 = ngllx * nglly * ngllz;
    Array<int> lex_ordering;
    if (lex_ref.Size() > 0) {
        lex_ordering.SetSize(lex_ref.Size());
        for (int i = 0; i < lex_ref.Size(); i++) {
            lex_ordering[i] = lex_ref[i];
        }
    } else {
        lex_ordering.SetSize(ngll3);
        for (int i = 0; i < ngll3; i++) lex_ordering[i] = i;
    }

    const int stride_k = nglly * ngllx;
    const int stride_e = ngllz * nglly * ngllx;
    Array<int> dofs;
    for (int e = 0; e < ne; e++) {
        fes.GetElementDofs(e, dofs);
        for (int k = 0; k < ngllz; k++) {
            for (int j = 0; j < nglly; j++) {
                for (int i = 0; i < ngllx; i++) {
                    int mat_idx = e * stride_e + k * stride_k + j * ngllx + i;
                    int lex_pos = i + j * ngllx + k * ngllx * nglly;
                    int internal_idx = lex_ordering[lex_pos];
                    int dof_idx = dofs[internal_idx];
                    if (dof_idx < 0) dof_idx = -1 - dof_idx;
                    mat_data[mat_idx] = gf_data[dof_idx];
                }
            }
        }
    }

    return field;
}


// =============================================================================
// ElasticMaterialBase2D Implementation
// =============================================================================

void ElasticMaterialBase2D::PrintInfo(std::ostream& os) const
{
    os << "ElasticMaterial2D:\n";
    os << "  Type: " << MaterialTypeToString(GetType()) << "\n";
    os << "  Elements: " << ne_ << "\n";
    os << "  GLL points: " << ngllx_ << " x " << nglly_ << "\n";
    os << "  Attenuation: " << (HasAttenuation() ? "enabled" : "disabled") << "\n";
    os << "  Memory: " << MemoryUsage() / 1024 << " KB\n";
}


// =============================================================================
// ElasticMaterialBase3D Implementation
// =============================================================================

void ElasticMaterialBase3D::PrintInfo(std::ostream& os) const
{
    os << "ElasticMaterial3D:\n";
    os << "  Type: " << MaterialTypeToString(GetType()) << "\n";
    os << "  Elements: " << ne_ << "\n";
    os << "  GLL points: " << ngllx_ << " x " << nglly_ << " x " << ngllz_ << "\n";
    os << "  Attenuation: " << (HasAttenuation() ? "enabled" : "disabled") << "\n";
    os << "  Memory: " << MemoryUsage() / 1024 << " KB\n";
}


// =============================================================================
// AcousticMaterialBase2D Implementation
// =============================================================================

void AcousticMaterialBase2D::PrintInfo(std::ostream& os) const
{
    os << "AcousticMaterial2D:\n";
    os << "  Type: " << MaterialTypeToString(GetType()) << "\n";
    os << "  Elements: " << ne_ << "\n";
    os << "  GLL points: " << ngllx_ << " x " << nglly_ << "\n";
    os << "  Attenuation: " << (HasAttenuation() ? "enabled" : "disabled") << "\n";
    os << "  Memory: " << MemoryUsage() / 1024 << " KB\n";
}


// =============================================================================
// AcousticMaterialBase3D Implementation
// =============================================================================

void AcousticMaterialBase3D::PrintInfo(std::ostream& os) const
{
    os << "AcousticMaterial3D:\n";
    os << "  Type: " << MaterialTypeToString(GetType()) << "\n";
    os << "  Elements: " << ne_ << "\n";
    os << "  GLL points: " << ngllx_ << " x " << nglly_ << " x " << ngllz_ << "\n";
    os << "  Attenuation: " << (HasAttenuation() ? "enabled" : "disabled") << "\n";
    os << "  Memory: " << MemoryUsage() / 1024 << " KB\n";
}

}  // namespace SEM
