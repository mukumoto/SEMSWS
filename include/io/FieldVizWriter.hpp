/**
 * @file FieldVizWriter.hpp
 * @brief Reusable scalar-field visualisation writers (paraview/glvis/gmt)
 *
 * Takes a pre-built list of (name, ParGridFunction*) pairs and emits
 * visualisation files. Used by:
 *   - MaterialWriter (runtime material output from a live simulation)
 *   - sem_viz        (offline tool converting .bp files to viz formats)
 *
 * All fields are assumed scalar (vdim == 1). Vector-field GMT output
 * with component selection lives in GMTWavefieldWriter.
 */

#ifndef SEM_FIELD_VIZ_WRITER_HPP
#define SEM_FIELD_VIZ_WRITER_HPP

#include "config/ConfigTypes.hpp"
#include <mfem.hpp>
#include <string>
#include <utility>
#include <vector>

namespace SEM {
namespace FieldVizWriter {

using namespace mfem;

using FieldList = std::vector<std::pair<std::string, ParGridFunction*>>;

/// Write a ParaView (.pvd + .vtu) dataset under `<output_dir>/<subdir>/paraview/`.
void WriteParaView(const std::string& output_dir,
                   const std::string& subdir,
                   const FieldList& fields,
                   const OutputFormatConfig& fmt,
                   ParMesh* mesh);

/// Write GLVis files (`<name>.gf` + shared `mesh.mesh`) under `<output_dir>/<subdir>/glvis/`.
void WriteGLVis(const std::string& output_dir,
                const std::string& subdir,
                const FieldList& fields);

/// Write GMT ASCII (.xyz) files under `<output_dir>/<subdir>/gmt/` (optionally per cross-section
/// subdirectory in 3D).
void WriteGMT(const std::string& output_dir,
              const std::string& subdir,
              const FieldList& fields,
              const OutputFormatConfig& fmt,
              ParMesh& mesh,
              MPI_Comm comm);

}  // namespace FieldVizWriter
}  // namespace SEM

#endif  // SEM_FIELD_VIZ_WRITER_HPP
