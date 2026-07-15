#pragma once

#include <string>
#include <mat.h>

ncnn::Mat load_image_to_ncnn_mat(const std::string& image_path);

ncnn::Mat ncnn_mat_resize(const ncnn::Mat& src, int target_w, int target_h);

ncnn::Mat ncnn_mat_resize_bicubic(const ncnn::Mat& src, int target_w, int target_h);

bool ncnn_mat_empty(const ncnn::Mat& mat);