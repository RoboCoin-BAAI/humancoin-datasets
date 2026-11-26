// Observational per-feature diagnostics. Never consulted by the estimator.
#ifndef OV_MSCKF_FEATURE_DIAGNOSTICS_H
#define OV_MSCKF_FEATURE_DIAGNOSTICS_H
#include <limits>
#include <map>
#include <cstddef>
#include <string>
#include "feat/FeatureInitializer.h"

namespace ov_msckf {
struct FeatureDiagnostic {
  std::string stage, outcome = "not_evaluated";
  size_t feature_id = 0, observations = 0;
  ov_core::FeatureInitializer::Diagnostics triangulation, refinement;
  double raw_pixel_rms = std::numeric_limits<double>::quiet_NaN();
  double raw_pixel_max_abs = std::numeric_limits<double>::quiet_NaN();
  double projected_residual_rms = std::numeric_limits<double>::quiet_NaN();
  double chi2 = std::numeric_limits<double>::quiet_NaN();
  double chi2_threshold = std::numeric_limits<double>::quiet_NaN();
  int residual_dof = 0;
};
}
#endif
