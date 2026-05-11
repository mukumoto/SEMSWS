/**
 * @file WavefieldWriterFactory.hpp
 * @brief Factory for creating wavefield writers from configuration
 */

#ifndef SEM_WAVEFIELD_WRITER_FACTORY_HPP
#define SEM_WAVEFIELD_WRITER_FACTORY_HPP

#include "io/WavefieldWriter.hpp"
#include "config/ConfigTypes.hpp"
#include <memory>
#include <vector>
#include <string>

namespace SEM {

/**
 * @brief Create wavefield writers from multi-format configuration
 * @param config Wavefield output configuration (may contain multiple formats)
 * @param output_dir Base output directory
 * @return Vector of wavefield writer instances
 */
std::vector<std::unique_ptr<WavefieldWriter>> CreateWavefieldWriters(
    const WavefieldOutputConfig& config,
    const std::string& output_dir);

}  // namespace SEM

#endif  // SEM_WAVEFIELD_WRITER_FACTORY_HPP
