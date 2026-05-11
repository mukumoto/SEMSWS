/**
 * @file PointFinder.hpp
 * @brief Custom point location for SEMSWS (replaces FindPointsGSLIB)
 *
 * This provides a precision-agnostic point finder that works with both
 * single and double precision MFEM builds, unlike GSLIB which is hardcoded
 * to double precision.
 *
 * NOTE: Internally uses double precision for Newton iteration accuracy,
 * but interfaces with MFEM's real_t for input/output compatibility.
 */

#ifndef SEM_POINT_FINDER_HPP
#define SEM_POINT_FINDER_HPP

#include "mfem.hpp"
#include <vector>

namespace SEM {


using namespace mfem;

/**
 * @struct PointLocation
 * @brief Result of point location search
 */
struct PointLocation {
    int elem = -1;              ///< Local element index (-1 if not found)
    int owner_rank = -1;        ///< MPI rank that owns the element
    int code = 2;               ///< 0=inside, 1=on boundary, 2=not found
    Vector ref_pos;             ///< Reference coordinates within element (real_t)

    PointLocation() : ref_pos(3) { ref_pos = 0.0; }
};

/**
 * @class PointFinder
 * @brief Locates physical points within a parallel mesh
 *
 * Algorithm:
 * 1. Each rank searches its local elements for the query points
 * 2. For each element, use Newton iteration to find reference coordinates
 * 3. Communicate results across all ranks to determine point ownership
 *
 * This implementation works with any precision (float or double) since
 * it uses MFEM's real_t throughout.
 */
class PointFinder {
public:
    /**
     * @brief Constructor
     * @param comm MPI communicator
     */
    explicit PointFinder(MPI_Comm comm);

    /**
     * @brief Setup the finder with a mesh
     * @param pmesh Parallel mesh to search in
     * @param newton_tol Tolerance for Newton iteration (default 1e-12)
     * @param newton_max_iter Max Newton iterations (default 50)
     */
    // Note: For single precision (float), use tolerance >= 1e-5
    //       For double precision, 1e-12 is appropriate
#ifdef MFEM_USE_SINGLE
    void Setup(ParMesh& pmesh, real_t newton_tol = 1e-5, int newton_max_iter = 50);
#else
    void Setup(ParMesh& pmesh, real_t newton_tol = 1e-12, int newton_max_iter = 50);
#endif

    /**
     * @brief Find multiple points in the mesh
     * @param points Point coordinates [dim * npts] in byVDIM ordering
     * @return Vector of PointLocation results
     *
     * After calling this, use GetElem(), GetProc(), GetCode(), GetReferencePosition()
     * to access results (compatible with FindPointsGSLIB interface).
     */
    void FindPoints(const Vector& points);

    // FindPointsGSLIB-compatible interface
    const Array<unsigned int>& GetElem() const { return elem_; }
    const Array<unsigned int>& GetProc() const { return proc_; }
    const Array<unsigned int>& GetCode() const { return code_; }
    const Vector& GetReferencePosition() const { return ref_pos_; }

private:
    /**
     * @brief Search local elements for a point
     * @param phys_pt Physical coordinates
     * @param[out] elem Local element index if found
     * @param[out] ref_pt Reference coordinates if found
     * @return true if point found in local mesh
     */
    bool SearchLocalMesh(const Vector& phys_pt, int& elem, Vector& ref_pt);

    /**
     * @brief Check if point is inside element using Newton iteration
     * @param elem Element index
     * @param phys_pt Physical coordinates
     * @param[out] ref_pt Reference coordinates
     * @return true if point is inside element (within tolerance)
     */
    bool NewtonSolve(int elem, const Vector& phys_pt, Vector& ref_pt);

    /**
     * @brief Check if reference point is within element bounds
     */
    bool IsInsideReferenceElement(const Vector& ref_pt, Geometry::Type geom) const;

    /**
     * @brief Get element bounding box for quick rejection test
     */
    void GetElementBBox(int elem, Vector& bb_min, Vector& bb_max);

    MPI_Comm comm_;
    int rank_;
    int num_procs_;

    ParMesh* pmesh_ = nullptr;
    int dim_ = 0;

    real_t newton_tol_ = 1e-12;
    int newton_max_iter_ = 50;

    // Element bounding boxes for quick rejection
    std::vector<Vector> elem_bb_min_;
    std::vector<Vector> elem_bb_max_;
    bool bbox_computed_ = false;

    // Results (FindPointsGSLIB-compatible)
    Array<unsigned int> elem_;
    Array<unsigned int> proc_;
    Array<unsigned int> code_;
    Vector ref_pos_;
};

}  // namespace SEM

#endif  // SEM_POINT_FINDER_HPP
