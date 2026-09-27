#pragma once

template <typename T> T clamp(T value, T minValue, T maxValue) {
  if (value < minValue) {
    return minValue;
  }
  if (value > maxValue) {
    return maxValue;
  }
  return value;
}

template <typename T> T maxValue(T a, T b) { return a > b ? a : b; }

template <typename T> T minValue(T a, T b) { return a < b ? a : b; }
