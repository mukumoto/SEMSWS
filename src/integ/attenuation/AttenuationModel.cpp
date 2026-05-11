/**
 * @file AttenuationModel.cpp
 * @brief Implementation of attenuation memory state classes
 */

#include "integ/attenuation/AttenuationModel.hpp"

namespace SEM {

// =============================================================================
// ViscoelasticMemory2D Implementation (packed arrays like 3D)
// =============================================================================

ViscoelasticMemory2D::ViscoelasticMemory2D(int ne, int ngllx, int nglly, int n_units)
    : ne_(ne), ngllx_(ngllx), nglly_(nglly), n_units_(n_units)
{
    const int64_t gll_size = (int64_t)ngllx * nglly * ne;
    const int64_t M_packed_size = 3 * gll_size * n_units;       // [3, ngllx, nglly, ne, n_units]
    const int64_t coeffs_packed_size = 4 * gll_size * n_units;  // [4, ngllx, nglly, ne, n_units]
    const int64_t strain_old_size = 3 * gll_size;               // [3, ngllx, nglly, ne]

    MFEM_VERIFY(M_packed_size <= INT_MAX,
                "ViscoelasticMemory2D M_packed size (" << M_packed_size << ") exceeds int32 limit.");
    MFEM_VERIFY(coeffs_packed_size <= INT_MAX,
                "ViscoelasticMemory2D coeffs_packed size (" << coeffs_packed_size << ") exceeds int32 limit.");
    MFEM_VERIFY(strain_old_size <= INT_MAX,
                "ViscoelasticMemory2D strain_old_packed size (" << strain_old_size << ") exceeds int32 limit.");

    // Packed memory variables: [3, ngllx, nglly, ne, n_units]
    // Components: 0=M1(bulk), 1=M2(d11), 2=M3(d12)
    M_packed.SetSize(static_cast<int>(M_packed_size));

    // Crank-Nicolson coefficients: [4, ngllx, nglly, ne, n_units]
    // Components: 0=alpha_kappa, 1=alpha_mu, 2=strain_coeff_kappa, 3=strain_coeff_mu
    coeffs_packed.SetSize(static_cast<int>(coeffs_packed_size));

    // Old strain values: [3, ngllx, nglly, ne]
    // Components: 0=ekk_old, 1=d11_old, 2=d12_old
    strain_old_packed.SetSize(static_cast<int>(strain_old_size));

    // Enable GPU memory
    EnableDevice();

    Reset();
}

void ViscoelasticMemory2D::Reset()
{
    M_packed = 0.0;
    strain_old_packed = 0.0;
}

size_t ViscoelasticMemory2D::MemoryUsage() const
{
    return (M_packed.Size() + coeffs_packed.Size() + strain_old_packed.Size()) * sizeof(real_t);
}


// =============================================================================
// ViscoelasticMemory3D Implementation
// =============================================================================

ViscoelasticMemory3D::ViscoelasticMemory3D(int ne, int ngllx, int nglly, int ngllz, int n_units)
    : ne_(ne), ngllx_(ngllx), nglly_(nglly), ngllz_(ngllz), n_units_(n_units)
{
    const int64_t total_size = (int64_t)ne * ngllz * nglly * ngllx * n_units;
    const int64_t gll_size = (int64_t)ne * ngllz * nglly * ngllx;
    const int64_t M_packed_size = 6 * total_size;
    const int64_t coeffs_packed_size = 4 * total_size;
    const int64_t strain_old_size = 6 * gll_size;

    if (M_packed_size > INT_MAX)
    {
        mfem::err << "ViscoelasticMemory3D::M_packed size (" << M_packed_size << ") exceeds int32 limit.\n"
                  << "Reduce elements per GPU (ne=" << ne << ") or increase GPU count.\n"
                  << "Maximum elements per GPU/CPU for NGLL (with n_units=" << n_units << "):\n"
                  << "  NGLL | MAX NE\n"
                  << "  -----|----------\n"
                  << "    5  |   954,437\n"
                  << "    6  |   552,336\n"
                  << "    7  |   347,826\n"
                  << "    8  |   233,016\n";
        MFEM_ABORT("");
    }

    MFEM_VERIFY(coeffs_packed_size <= INT_MAX,
                "ViscoelasticMemory3D::coeffs_packed size (" << coeffs_packed_size << ") exceeds int32 limit.");
    MFEM_VERIFY(strain_old_size <= INT_MAX,
                "ViscoelasticMemory3D::strain_old_packed size (" << strain_old_size << ") exceeds int32 limit.");

    // Packed memory variables: [6, ngllx, nglly, ngllz, ne, n_units]
    M_packed.SetSize(static_cast<int>(M_packed_size));

    // Crank-Nicolson coefficients: [4, ngllx, nglly, ngllz, ne, n_units]
    coeffs_packed.SetSize(static_cast<int>(coeffs_packed_size));

    // Packed old strain: [6, ngllx, nglly, ngllz, ne]
    strain_old_packed.SetSize(static_cast<int>(strain_old_size));

    // Enable GPU memory
    M_packed.UseDevice(true);
    coeffs_packed.UseDevice(true);
    strain_old_packed.UseDevice(true);

    Reset();
}

void ViscoelasticMemory3D::Reset()
{
    M_packed = 0.0;
    strain_old_packed = 0.0;
}

size_t ViscoelasticMemory3D::MemoryUsage() const
{
    return (M_packed.Size() + coeffs_packed.Size() + strain_old_packed.Size()) * sizeof(real_t);
}


// =============================================================================
// ViscoacousticMemory2D Implementation (Crank-Nicolson scheme)
// =============================================================================

ViscoacousticMemory2D::ViscoacousticMemory2D(int ne, int ngllx, int nglly, int n_units)
    : ne_(ne), ngllx_(ngllx), nglly_(nglly), n_units_(n_units)
{
    const int64_t mem_size = (int64_t)ngllx * nglly * ne * n_units;
    const int64_t gll_size = (int64_t)ngllx * nglly * ne;
    const int64_t coeffs_size = 2 * mem_size;  // [2, ngllx, nglly, ne, n_units] for alpha, strain_coeff

    MFEM_VERIFY(mem_size <= INT_MAX,
                "ViscoacousticMemory2D e1 size (" << mem_size << ") exceeds int32 limit.");
    MFEM_VERIFY(gll_size <= INT_MAX,
                "ViscoacousticMemory2D dot_e1_old size (" << gll_size << ") exceeds int32 limit.");
    MFEM_VERIFY(coeffs_size <= INT_MAX,
                "ViscoacousticMemory2D coeffs_packed size (" << coeffs_size << ") exceeds int32 limit.");

    // Memory variable e1 [ngllx * nglly * ne * n_units]
    e1.SetSize(static_cast<int>(mem_size));

    // Crank-Nicolson coefficients [2, ngllx * nglly * ne * n_units]
    // Components: 0=alpha, 1=strain_coeff
    coeffs_packed.SetSize(static_cast<int>(coeffs_size));

    // Old stiffness output [ngllx * nglly * ne]
    dot_e1_old.SetSize(static_cast<int>(gll_size));

    // Enable GPU memory
    EnableDevice();

    Reset();
}

void ViscoacousticMemory2D::Reset()
{
    e1 = 0.0;
    dot_e1_old = 0.0;
}

size_t ViscoacousticMemory2D::MemoryUsage() const
{
    return (e1.Size() + coeffs_packed.Size() + dot_e1_old.Size()) * sizeof(real_t);
}


// =============================================================================
// ViscoacousticMemory3D Implementation (Crank-Nicolson scheme)
// =============================================================================

ViscoacousticMemory3D::ViscoacousticMemory3D(int ne, int ngllx, int nglly, int ngllz, int n_units)
    : ne_(ne), ngllx_(ngllx), nglly_(nglly), ngllz_(ngllz), n_units_(n_units)
{
    const int64_t mem_size = (int64_t)ngllx * nglly * ngllz * ne * n_units;
    const int64_t gll_size = (int64_t)ngllx * nglly * ngllz * ne;
    const int64_t coeffs_size = 2 * mem_size;  // [2, ngllx, nglly, ngllz, ne, n_units] for alpha, strain_coeff
    MFEM_VERIFY(mem_size <= INT_MAX,
                "ViscoacousticMemory3D e1 size (" << mem_size << ") exceeds int32 limit.");
    MFEM_VERIFY(gll_size <= INT_MAX,
                "ViscoacousticMemory3D dot_e1_old size (" << gll_size << ") exceeds int32 limit.");
    MFEM_VERIFY(coeffs_size <= INT_MAX,
                "ViscoacousticMemory3D coeffs_packed size (" << coeffs_size << ") exceeds int32 limit.");

    // Memory variable e1 [ngllx * nglly * ngllz * ne * n_units]
    e1.SetSize(static_cast<int>(mem_size));

    // Crank-Nicolson coefficients [2, ngllx * nglly * ngllz * ne * n_units]
    // Components: 0=alpha, 1=strain_coeff
    coeffs_packed.SetSize(static_cast<int>(coeffs_size));

    // Old stiffness output [ngllx * nglly * ngllz * ne]
    dot_e1_old.SetSize(static_cast<int>(gll_size));

    // Enable GPU memory
    EnableDevice();

    Reset();
}

void ViscoacousticMemory3D::Reset()
{
    e1 = 0.0;
    dot_e1_old = 0.0;
}

size_t ViscoacousticMemory3D::MemoryUsage() const
{
    return (e1.Size() + coeffs_packed.Size() + dot_e1_old.Size()) * sizeof(real_t);
}


}  // namespace SEM
