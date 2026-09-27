#include "app.h"
#include <cstring>
#include <filesystem>

int main(int argc, char **argv) {
  const std::filesystem::path appDir =
      std::filesystem::absolute(argv[0]).parent_path();

  bool pathTracing = false;
  for (int i = 1; i < argc; ++i) {
    pathTracing |= std::strcmp(argv[i], "--pathtrace") == 0;
  }

  App app(appDir);
  if (pathTracing) {
    app.setRenderMode(RenderMode::PathTracing);
  }
  app.run();
  return EXIT_SUCCESS;
}
