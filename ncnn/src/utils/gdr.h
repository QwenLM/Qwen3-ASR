#pragma once

#include <vector>
#include <mat.h>
#include <net.h>

class GatedDeltaRule : public ncnn::Layer
{
public:
    GatedDeltaRule();

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;

public:
    int num_k_heads;
    int num_v_heads;
};

class ShortConv : public ncnn::Layer
{
public:
    ShortConv();

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
};

void register_gdr_layers(ncnn::Net& net);