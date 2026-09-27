#include "app.h"

#include "renderer.h"

#include <GLFW/glfw3.h>

#include <glm/gtc/quaternion.hpp>

#include <chrono>
#include <iostream>

constexpr uint32_t kWidth = 1920 * 1.5;
constexpr uint32_t kHeight = 1080 * 1.5;

App::App(std::filesystem::path appDir) : _resources(appDir) {

  initWindow();
  _renderer = std::make_unique<Renderer>(_window, appDir);
  _renderer->init();
  initScene();
}

App::~App() {
  _renderer.reset();
  if (_window) {
    glfwDestroyWindow(_window);
    _window = nullptr;
  }
  glfwTerminate();
}

void App::initWindow() {
  const int glfwInitialized = glfwInit();
  assert(glfwInitialized && "failed to initialize GLFW");

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  _window = glfwCreateWindow(kWidth, kHeight, "vktest", nullptr, nullptr);
  assert(_window && "failed to create GLFW window");
}

void App::initScene() {
  struct SceneModel {
    const char *path;
    glm::vec3 position;
  };
  // each model is scaled to a bounding radius of 1 and placed on the X axis
  const SceneModel sceneModels[] = {
      {"assets/DamagedHelmet.glb", {0.0f, 0.0f, 0.0f}},
      {"assets/Duck.glb", {2.5f, 0.0f, 0.0f}},
      {"assets/kitten.glb", {-2.5f, 0.0f, 0.0f}},
  };

  for (const SceneModel &sceneModel : sceneModels) {
    const GltfModel model = _resources.loadGltfModel(sceneModel.path);
    std::cout << sceneModel.path << ": " << model.meshlets.size()
              << " meshlets, " << model.meshletTriangles.size()
              << " triangles, "
              << model.materials.size() << " materials, "
              << model.textures.size() << " textures" << std::endl;

    Object3D object(_renderer->loadModel(model));
    const float scale =
        model.boundsRadius > 0.0f ? 1.0f / model.boundsRadius : 1.0f;
    object.scale = glm::vec3(scale);
    object.position = sceneModel.position - model.boundsCenter * scale;
    _objects.push_back(object);
  }

  _renderer->setEnvironment(
      _resources.loadHdrImage("assets/photo_studio_loft_hall_2k.hdr"));

  _camera.lookAt(glm::vec3{0.0f, 0.5f, 5.0f}, glm::vec3{0.0f});
  _renderer->setObjects(_objects);
}

void App::setRenderMode(RenderMode mode) { _renderer->setRenderMode(mode); }

void App::run() {
  auto previousTime = std::chrono::steady_clock::now();

  while (!glfwWindowShouldClose(_window)) {
    const auto now = std::chrono::steady_clock::now();
    const float dt =
        std::chrono::duration<float, std::milli>(now - previousTime).count();
    previousTime = now;

    _camera.update(_window, dt);
    _renderer->render(_camera.viewProjectionMatrix(), _camera.cullData(), dt);
    glfwPollEvents();
  }
  _renderer->waitIdle();
}
