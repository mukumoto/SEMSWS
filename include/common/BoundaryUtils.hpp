/**
 * @file BoundaryUtils.hpp
 * @brief Utility functions for boundary handling
 */

#ifndef SEM_COMMON_BOUNDARY_UTILS_HPP
#define SEM_COMMON_BOUNDARY_UTILS_HPP

#include <string>
#include <stdexcept>
#include <mfem.hpp>

namespace SEM {

/**
 * @brief Parse boundary side string to MFEM boundary attribute
 *
 * Maps human-readable side names to MFEM boundary attributes for box meshes.
 *
 * 2D mapping (quad mesh):
 *   - "bottom" -> 1
 *   - "right"  -> 2
 *   - "top"    -> 3
 *   - "left"   -> 4
 *
 * 3D mapping (hex mesh, matches MFEM MakeCartesian3D):
 *   - "bottom" -> 1 (z = z_min)
 *   - "front"  -> 2 (y = y_min)
 *   - "right"  -> 3 (x = x_max)
 *   - "back"   -> 4 (y = y_max)
 *   - "left"   -> 5 (x = x_min)
 *   - "top"    -> 6 (z = z_max)
 *
 * @param side Side name (case-sensitive)
 * @param dim Spatial dimension (2 or 3)
 * @return Boundary attribute number, or -1 if invalid
 */
inline int ParseBoundarySide(const std::string& side, int dim) {
    if (dim == 2) {
        if (side == "bottom") return 1;
        if (side == "right") return 2;
        if (side == "top") return 3;
        if (side == "left") return 4;
    } else {
        // 3D: matches MFEM MakeCartesian3D boundary attributes
        if (side == "bottom") return 1;  // z = z_min
        if (side == "front") return 2;   // y = y_min
        if (side == "right") return 3;   // x = x_max
        if (side == "back") return 4;    // y = y_max
        if (side == "left") return 5;    // x = x_min
        if (side == "top") return 6;     // z = z_max
    }
    return -1;
}

/**
 * @brief Get boundary attribute name from attribute number
 *
 * Inverse of ParseBoundarySide.
 *
 * @param attr Boundary attribute number
 * @param dim Spatial dimension (2 or 3)
 * @return Side name, or empty string if invalid
 */
inline std::string GetBoundarySideName(int attr, int dim) {
    if (dim == 2) {
        switch (attr) {
            case 1: return "bottom";
            case 2: return "right";
            case 3: return "top";
            case 4: return "left";
        }
    } else {
        // 3D: matches MFEM MakeCartesian3D boundary attributes
        switch (attr) {
            case 1: return "bottom";  // z = z_min
            case 2: return "front";   // y = y_min
            case 3: return "right";   // x = x_max
            case 4: return "back";    // y = y_max
            case 5: return "left";    // x = x_min
            case 6: return "top";     // z = z_max
        }
    }
    return "";
}

/**
 * @brief Parse boundary attribute value from string (named side or integer)
 *
 * Accepts either a named side ("top", "left", etc.) or a positive integer
 * string ("5", "7", etc.) for external mesh boundary attributes.
 *
 * @param value String value from config (e.g., "top" or "5")
 * @param dim Spatial dimension (2 or 3)
 * @return Boundary attribute number (1-based), or -1 if invalid
 */
inline int ParseBoundaryAttributeValue(const std::string& value, int dim) {
    // Try named side first
    int attr = ParseBoundarySide(value, dim);
    if (attr > 0) return attr;

    // Try integer
    try {
        int num = std::stoi(value);
        if (num > 0) return num;
    } catch (const std::invalid_argument&) {
    } catch (const std::out_of_range&) {
    }

    return -1;
}

}  // namespace SEM

#endif  // SEM_COMMON_BOUNDARY_UTILS_HPP
