#include "app.h"
#include <cstdlib>
#include <cstring>
#include <filesystem>

int main(int argc, char **argv) {
  const std::filesystem::path appDir =
      std::filesystem::absolute(argv[0]).parent_path();

  bool pathTracing = false;
  bool environmentSampling = true;
  uint32_t maxSamples = 0;
  std::filesystem::path dumpPath;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--pathtrace") == 0) {
      pathTracing = true;
    } else if (std::strcmp(argv[i], "--no-env-sampling") == 0) {
      environmentSampling = false;
    } else if (std::strcmp(argv[i], "--max-samples") == 0 && i + 1 < argc) {
      maxSamples = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
      dumpPath = argv[++i];
    }
  }

  App app(appDir);
  if (pathTracing) {
    app.setRenderMode(RenderMode::PathTracing);
  }
  app.renderer().setEnvironmentSampling(environmentSampling);
  app.renderer().setMaxSamples(maxSamples);
  app.setDumpPath(dumpPath);
  app.run();
  return EXIT_SUCCESS;
}
