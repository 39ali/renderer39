#pragma once

#include "camera.h"
#include "object3d.h"
#include "renderer.h"
#include "resource.h"

#include <memory>
#include <vector>

struct GLFWwindow;
class Renderer;

class App {
public:
  // scene: a single glTF to show instead of the default models (optional)
  App(std::filesystem::path appDir, std::filesystem::path scene = {});
  ~App();
  void run();
  void setRenderMode(RenderMode mode);
  Renderer &renderer() { return *_renderer; }
  // saves the path tracer's result and quits once it reaches its sample limit
  void setDumpPath(std::filesystem::path path) { _dumpPath = std::move(path); }

private:
  void initWindow();
  void initScene(const std::filesystem::path &scene);

  Resources _resources;
  std::vector<Object3D> _objects;
  Camera _camera;
  GLFWwindow *_window = nullptr;
  std::unique_ptr<Renderer> _renderer;
  std::filesystem::path _dumpPath;
};
