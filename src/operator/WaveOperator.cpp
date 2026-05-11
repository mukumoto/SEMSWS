/**
 * @file WaveOperator.cpp
 * @brief Base class implementation for wave operators
 */

#include "operator/WaveOperator.hpp"

namespace SEM {

WaveOperator::WaveOperator(ParFiniteElementSpace& fes)
    : SecondOrderTimeDependentOperator(fes.GetTrueVSize(), real_t(0.0)),
      fes_(fes) {}

void WaveOperator::PrintInfo(std::ostream& os) const {
    os << "WaveOperator:\n";
    os << "Local  DOFs: " << fes_.GetTrueVSize() << "\n";
    os << "Local  Elements: " << fes_.GetParMesh()->GetNE() << "\n";
}

}  // namespace SEM
