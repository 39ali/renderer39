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
  App(std::filesystem::path appDir);
  ~App();
  void run();
  void setRenderMode(RenderMode mode);

private:
  void initWindow();
  void initScene();

  Resources _resources;
  std::vector<Object3D> _objects;
  Camera _camera;
  GLFWwindow *_window = nullptr;
  std::unique_ptr<Renderer> _renderer;
};
