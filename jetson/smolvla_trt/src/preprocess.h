// The camera frame as SmolVLA's vision encoder gets it (SmolVLAPolicy.prepare_images):
// scaled to fit 512x512 keeping its aspect ratio (bilinear, as F.interpolate with
// align_corners=False), padded on the left and top, and mapped from [0, 1] to [-1, 1].
// A 320x240 camera becomes 512x384 with 128 rows of padding on top: the top two rows of
// the 8x8 grid of image tokens see only padding.
#pragma once

#include <opencv2/core.hpp>

#include <vector>

struct ModelImage {
  std::vector<float> chw;        // [3, H, W], values in [-1, 1]; padding is -1
  int width = 0, height = 0;     // the model's input (512 x 512)
  int source_w = 0, source_h = 0;
  int resized_w = 0, resized_h = 0;
  int pad_left = 0, pad_top = 0;
  cv::Mat rgb;                   // the same as uint8 RGB, for looking at
};

ModelImage prepare_image(const cv::Mat& rgb, int width, int height);

// The model input with the token grid drawn on it, padding hatched, and (optionally) each
// token's cell tinted by `cell_values` (grid*grid values, e.g. image-token norms). BGR.
cv::Mat draw_token_grid(const ModelImage& image, int grid, const std::vector<float>& cell_values = {});
