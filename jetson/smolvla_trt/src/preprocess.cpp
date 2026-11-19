#include "preprocess.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

ModelImage prepare_image(const cv::Mat& rgb, int width, int height) {
  if (rgb.empty() || rgb.type() != CV_8UC3) throw std::runtime_error("expected an 8-bit, 3-channel RGB image");
  ModelImage m;
  m.width = width, m.height = height, m.source_w = rgb.cols, m.source_h = rgb.rows;

  // resize_with_pad: ratio = max(w / W, h / H); sizes truncated as Python's int() does.
  double ratio = std::max(static_cast<double>(rgb.cols) / width, static_cast<double>(rgb.rows) / height);
  m.resized_w = static_cast<int>(rgb.cols / ratio);
  m.resized_h = static_cast<int>(rgb.rows / ratio);
  m.pad_left = std::max(0, width - m.resized_w);
  m.pad_top = std::max(0, height - m.resized_h);

  cv::Mat unit, resized;
  rgb.convertTo(unit, CV_32FC3, 1.0 / 255.0);  // in float, as LeRobot does before resizing
  if (m.resized_w == rgb.cols && m.resized_h == rgb.rows) resized = unit;
  else cv::resize(unit, resized, cv::Size(m.resized_w, m.resized_h), 0, 0, cv::INTER_LINEAR);

  m.chw.assign(static_cast<size_t>(3) * width * height, -1.0f);  // padding: 0 in [0, 1] -> -1
  for (int y = 0; y < m.resized_h; ++y) {
    const auto* row = resized.ptr<cv::Vec3f>(y);
    for (int x = 0; x < m.resized_w; ++x)
      for (int c = 0; c < 3; ++c)
        m.chw[(static_cast<size_t>(c) * height + (y + m.pad_top)) * width + (x + m.pad_left)] = row[x][c] * 2.0f - 1.0f;
  }
  m.rgb = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat view;
  resized.convertTo(view, CV_8UC3, 255.0);
  view.copyTo(m.rgb(cv::Rect(m.pad_left, m.pad_top, m.resized_w, m.resized_h)));
  return m;
}

cv::Mat draw_token_grid(const ModelImage& image, int grid, const std::vector<float>& cell_values) {
  cv::Mat bgr;
  cv::cvtColor(image.rgb, bgr, cv::COLOR_RGB2BGR);
  const int cw = image.width / grid, ch = image.height / grid;
  float lo = 0, hi = 1;
  if (!cell_values.empty()) {
    lo = *std::min_element(cell_values.begin(), cell_values.end());
    hi = *std::max_element(cell_values.begin(), cell_values.end());
  }
  for (int r = 0; r < grid; ++r) {
    for (int c = 0; c < grid; ++c) {
      cv::Rect cell(c * cw, r * ch, cw, ch);
      bool padding = cell.y + cell.height <= image.pad_top || cell.x + cell.width <= image.pad_left;
      if (padding) {  // hatch: these tokens see nothing but padding
        for (int k = -ch; k < cw; k += 8)
          cv::line(bgr, {cell.x + k, cell.y + ch}, {cell.x + k + ch, cell.y}, cv::Scalar(70, 70, 70), 1);
      }
      if (!cell_values.empty()) {
        float v = (cell_values[r * grid + c] - lo) / std::max(hi - lo, 1e-6f);
        cv::Mat roi = bgr(cell);
        cv::Mat tint(roi.size(), roi.type(), cv::Scalar(255 * (1 - v), 64, 255 * v));
        cv::addWeighted(roi, 0.6, tint, 0.4, 0, roi);
      }
      cv::rectangle(bgr, cell, cv::Scalar(255, 255, 0), 1);
      cv::putText(bgr, std::to_string(r * grid + c), {cell.x + 3, cell.y + 14}, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
  }
  return bgr;
}
