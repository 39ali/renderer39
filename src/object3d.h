#pragma once

#include "resource.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class Object3D {
public:
  Object3D(MeshId meshId) : meshId(meshId) {}

  MeshId meshId = 0;
  glm::vec3 position{0.0f};
  glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec3 scale{1.0f};
};

