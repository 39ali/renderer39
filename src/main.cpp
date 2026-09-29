#include "app.h"
#include <cstdlib>
#include <cstring>
#include <filesystem>

int main(int argc, char **argv) {
  const std::filesystem::path appDir =
      std::filesystem::absolute(argv[0]).parent_path();

  bool pathTracing = false;
  bool environmentSampling = true;
  bool emissiveSampling = true;
  float lightIntensity = -1.0f;
  float environmentIntensity = -1.0f;
  float emissiveIntensity = -1.0f;
  uint32_t maxSamples = 0;
  std::filesystem::path dumpPath;
  std::filesystem::path scene;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--pathtrace") == 0) {
      pathTracing = true;
    } else if (std::strcmp(argv[i], "--no-env-sampling") == 0) {
      environmentSampling = false;
    } else if (std::strcmp(argv[i], "--max-samples") == 0 && i + 1 < argc) {
      maxSamples = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--no-emissive-sampling") == 0) {
      emissiveSampling = false;
    } else if (std::strcmp(argv[i], "--light-intensity") == 0 &&
               i + 1 < argc) {
      lightIntensity = std::strtof(argv[++i], nullptr);
    } else if (std::strcmp(argv[i], "--emissive-intensity") == 0 &&
               i + 1 < argc) {
      emissiveIntensity = std::strtof(argv[++i], nullptr);
    } else if (std::strcmp(argv[i], "--env-intensity") == 0 && i + 1 < argc) {
      environmentIntensity = std::strtof(argv[++i], nullptr);
    } else if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
      scene = argv[++i];
    } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
      dumpPath = argv[++i];
    }
  }

  App app(appDir, scene);
  if (pathTracing) {
    app.setRenderMode(RenderMode::PathTracing);
  }
  app.renderer().setEnvironmentSampling(environmentSampling);
  app.renderer().setMaxSamples(maxSamples);
  app.renderer().setEmissiveSampling(emissiveSampling);
  if (lightIntensity >= 0.0f) {
    app.renderer().setLightIntensity(lightIntensity);
  }
  if (emissiveIntensity >= 0.0f) {
    app.renderer().setEmissiveIntensity(emissiveIntensity);
  }
  if (environmentIntensity >= 0.0f) {
    app.renderer().setEnvironmentIntensity(environmentIntensity);
  }
  app.setDumpPath(dumpPath);
  app.run();
  return EXIT_SUCCESS;
}
