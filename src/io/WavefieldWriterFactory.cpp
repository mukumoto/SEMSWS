/**
 * @file WavefieldWriterFactory.cpp
 * @brief Implementation of wavefield writer factory
 */

#include "io/WavefieldWriterFactory.hpp"
#include "io/WavefieldWriter.hpp"
#include "io/ParaViewWavefieldWriter.hpp"
#include "io/GMTWavefieldWriter.hpp"

#include <algorithm>

namespace SEM {

namespace {

/// Check if a field name is in the fields list
bool HasField(const std::vector<std::string>& fields, const std::string& name) {
    return std::find(fields.begin(), fields.end(), name) != fields.end();
}

}  // anonymous namespace

std::vector<std::unique_ptr<WavefieldWriter>> CreateWavefieldWriters(
    const WavefieldOutputConfig& config,
    const std::string& output_dir) {

    std::vector<std::unique_ptr<WavefieldWriter>> writers;

    for (const auto& fmt : config.formats) {
        if (fmt.type == "glvis") {
            // GLVis writer (existing, only writes displacement)
            writers.push_back(
                std::make_unique<GLVisWavefieldWriter>(output_dir, config.interval));

        } else if (fmt.type == "paraview") {
            ParaViewWavefieldWriter::Options opts;
            opts.refinement = fmt.refinement;
            opts.data_format = fmt.data_format;
            opts.compression = fmt.compression;
            opts.write_displacement = HasField(config.fields, "DISP");
            opts.write_velocity = HasField(config.fields, "VEL");
            opts.write_acceleration = HasField(config.fields, "ACC");
            opts.write_pressure = HasField(config.fields, "PS");

            writers.push_back(
                std::make_unique<ParaViewWavefieldWriter>(
                    output_dir, config.interval, opts));

        } else if (fmt.type == "gmt") {
            GMTWavefieldWriter::Options opts;
            opts.resolution = fmt.resolution;
            opts.components = fmt.components;
            opts.cross_sections = fmt.cross_sections;
            opts.write_displacement = HasField(config.fields, "DISP");
            opts.write_velocity = HasField(config.fields, "VEL");
            opts.write_acceleration = HasField(config.fields, "ACC");
            opts.write_pressure = HasField(config.fields, "PS");

            writers.push_back(
                std::make_unique<GMTWavefieldWriter>(
                    output_dir, config.interval, opts));

        } else {
            MFEM_ABORT("Unknown wavefield output format: " + fmt.type);
        }
    }

    return writers;
}

}  // namespace SEM
