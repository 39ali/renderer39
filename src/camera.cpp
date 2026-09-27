#include "camera.h"
#include "math_helper.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

void Camera::update(GLFWwindow *window, float dt) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS && _mouseLook) {
    _mouseLook = false;
    _hasLastMousePosition = false;
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  }

  const bool wantsMouseLook =
      glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
  if (wantsMouseLook != _mouseLook) {
    _mouseLook = wantsMouseLook;
    _hasLastMousePosition = false;
    glfwSetInputMode(window, GLFW_CURSOR,
                     _mouseLook ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
  }

  double mouseX = 0.0;
  double mouseY = 0.0;
  glfwGetCursorPos(window, &mouseX, &mouseY);
  onCursorPosition(mouseX, mouseY);

  auto right = this->right();
  auto forward = this->forward();

  const float speed = _moveSpeed * (dt / 1000.0f);
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    _position += forward * speed;
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    _position -= forward * speed;
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    _position += right * speed;
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    _position -= right * speed;
  }
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
    _position.y += speed;
  }
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
    _position.y -= speed;
  }

  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  if (height > 0) {
    _aspect = static_cast<float>(width) / static_cast<float>(height);
  }
}

void Camera::onCursorPosition(double x, double y) {
  if (!_mouseLook) {
    return;
  }

  if (!_hasLastMousePosition) {
    _lastMouseX = x;
    _lastMouseY = y;
    _hasLastMousePosition = true;
    return;
  }

  _yaw += static_cast<float>(x - _lastMouseX) * _mouseSensitivity;
  _pitch += static_cast<float>(y - _lastMouseY) * _mouseSensitivity;
  _pitch = clamp(_pitch, -89.0f, 89.0f);
  _lastMouseX = x;
  _lastMouseY = y;
}

void Camera::lookAt(const glm::vec3 &position, const glm::vec3 &target) {
  _position = position;
  const glm::vec3 direction = glm::normalize(target - position);
  _yaw = glm::degrees(std::atan2(direction.z, direction.x));
  _pitch = glm::degrees(std::asin(clamp(direction.y, -1.0f, 1.0f)));
  _hasLastMousePosition = false;
}

glm::mat4 Camera::viewMatrix() const {
  return glm::lookAt(_position, _position + forward(),
                     glm::vec3{0.0f, 1.0f, 0.0f});
}

glm::mat4 Camera::projectionMatrix() const {
  glm::mat4 projection =
      glm::perspective(glm::radians(_fovY), _aspect, _nearPlane, _farPlane);
  // Vulkan clip space has Y pointing down
  projection[1][1] *= -1.0f;
  return projection;
}

glm::mat4 Camera::viewProjectionMatrix() const {
  return projectionMatrix() * viewMatrix();
}

CameraCullData Camera::cullData() const {
  const float halfFovY = glm::radians(_fovY) * 0.5f;
  const float halfFovX = std::atan(std::tan(halfFovY) * _aspect);
  return {
      .view = viewMatrix(),
      .frustum = {std::cos(halfFovX), std::sin(halfFovX),
                  std::cos(halfFovY), std::sin(halfFovY)},
      .zNearFar = {_nearPlane, _farPlane},
  };
}

glm::vec3 Camera::forward() const {
  const float yaw = glm::radians(_yaw);
  const float pitch = glm::radians(_pitch);
  return glm::normalize(glm::vec3{
      std::cos(yaw) * std::cos(pitch),
      std::sin(pitch),
      std::sin(yaw) * std::cos(pitch),
  });
}

glm::vec3 Camera::right() const {
  return glm::normalize(glm::cross(forward(), glm::vec3{0.0f, 1.0f, 0.0f}));
}
