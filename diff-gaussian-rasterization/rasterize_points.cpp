/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <memory>
#include <filesystem>
extern "C"
{
#include "cpu_rasterizer/config.h"
#include "cpu_rasterizer/rasterizer.h"
}
#include <fstream>
#include <string>
#include <functional>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
  auto lambda = [&t](size_t N)
  {
    t.resize_({(long long)N});
    return reinterpret_cast<char *>(t.contiguous().data_ptr());
  };
  return lambda;
}

// 定义 dump_tensor 函数
void dump_tensor(std::fstream &file, const torch::Tensor &tensor)
{
  // 获取张量的数据类型
  torch::ScalarType dtype = tensor.scalar_type();

  // 根据数据类型确定每个元素的字节大小
  size_t element_size;
  switch (dtype)
  {
  case torch::kFloat32:
    element_size = sizeof(float);
    break;
  case torch::kInt32:
    element_size = sizeof(int);
    break;
  // 根据需要添加更多数据类型
  default:
    throw std::runtime_error("Unsupported tensor data type");
  }

  // 计算总字节数
  size_t total_size = tensor.numel() * element_size;

  // 写入数据
  file.write(reinterpret_cast<const char *>(tensor.contiguous().data_ptr()), total_size);

  if (!file)
  {
    throw std::runtime_error("Failed to write tensor data to file");
  }
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCPU(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &colors,
    const torch::Tensor &opacity,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const bool prefiltered,
    const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3)
  {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCPU);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  // std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  // std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  // std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0)
  {
    int M = 0;
    if (sh.size(0) != 0)
    {
      M = sh.size(1);
    }

    // 在写文件之前创建目录
    std::filesystem::path dump_dir("diff-gaussian-rasterization/data_dump");
    if (!std::filesystem::exists(dump_dir))
    {
      std::filesystem::create_directories(dump_dir);
    }

    std::fstream dump_file("diff-gaussian-rasterization/data_dump/rasterizer_in.bin", std::ios::binary | std::ios::out | std::ios::in | std::ios::trunc);

    dump_file.write(reinterpret_cast<const char *>(&P), sizeof(int));
    // // 检查写入文件的P是否和原P相等
    // // 读取写入的P值进行验证
    // int written_P;
    // dump_file.seekg(0);
    // dump_file.read(reinterpret_cast<char *>(&written_P), sizeof(int));
    // dump_file.seekp(sizeof(int)); // 重置写入位置
    // if (written_P != P)
    // {
    //   std::cout << "写入的P值(" << written_P << ")与原始P值(" << P << ")不匹配!" << std::endl;
    // }
    dump_file.write(reinterpret_cast<const char *>(&degree), sizeof(int));
    dump_file.write(reinterpret_cast<const char *>(&M), sizeof(int));
    dump_file.write(reinterpret_cast<const char *>(&W), sizeof(int));
    dump_file.write(reinterpret_cast<const char *>(&H), sizeof(int));
    dump_file.write(reinterpret_cast<const char *>(&scale_modifier), sizeof(float));
    dump_file.write(reinterpret_cast<const char *>(&tan_fovx), sizeof(float));
    dump_file.write(reinterpret_cast<const char *>(&tan_fovy), sizeof(float));
    dump_file.write(reinterpret_cast<const char *>(&prefiltered), sizeof(bool));
    dump_file.write(reinterpret_cast<const char *>(&debug), sizeof(bool));
    // std::cout << "P: " << P << std::endl;
    // std::cout << "degree: " << degree << std::endl;
    // std::cout << "M: " << M << std::endl;
    // std::cout << "W: " << W << std::endl;
    // std::cout << "H: " << H << std::endl;
    // std::cout << "scale_modifier: " << scale_modifier << std::endl;
    // std::cout << "tan_fovx: " << tan_fovx << std::endl;
    // std::cout << "tan_fovy: " << tan_fovy << std::endl;
    // std::cout << "prefiltered: " << prefiltered << std::endl;
    // std::cout << "debug: " << debug << std::endl;

    dump_tensor(dump_file, background);
    // 记录当前写入位置
    std::streampos current_pos = dump_file.tellp();
    std::streampos background_pos = current_pos - static_cast<std::streamoff>(background.numel() * sizeof(float));

    // 读取写入的 background 数据
    torch::Tensor read_background = torch::empty_like(background);
    dump_file.seekg(background_pos);
    dump_file.read(reinterpret_cast<char *>(read_background.data_ptr()), background.numel() * sizeof(float));
    for (int i = 0; i < background.numel(); i++)
    {
      std::cout << "background[" << i << "]: " << background.data_ptr<float>()[i] << std::endl;
    }
    dump_tensor(dump_file, means3D);
    for (int i = 0; i < 10; i++)
    {
      std::cout << "means3D[" << i << "]: " << means3D.data_ptr<float>()[i] << std::endl;
    }
    dump_tensor(dump_file, sh);
    for (int i = 0; i < 10; i++)
    {
      std::cout << "sh[" << i << "]: " << sh.data_ptr<float>()[i] << std::endl;
    }
    dump_tensor(dump_file, colors);
    dump_tensor(dump_file, opacity);
    for (int i = 0; i < 10; i++)
    {
      std::cout << "opacity[" << i << "]: " << opacity.data_ptr<float>()[i] << std::endl;
    }
    dump_tensor(dump_file, scales);
    for (int i = 0; i < 10; i++)
    {
      std::cout << "scales[" << i << "]: " << scales.data_ptr<float>()[i] << std::endl;
    }
    dump_tensor(dump_file, rotations);
    for (int i = 0; i < 10; i++)
    {
      std::cout << "rotations[" << i << "]: " << rotations.data_ptr<float>()[i] << std::endl;
    }
    dump_tensor(dump_file, cov3D_precomp);
    std::cout << "viewmatrix shape: " << viewmatrix.sizes() << std::endl;
    std::cout << "viewmatrix stride: " << viewmatrix.strides() << std::endl;
    std::cout << "viewmatrix is_contiguous: " << viewmatrix.is_contiguous() << std::endl;
    dump_tensor(dump_file, viewmatrix);
    for (int i = 0; i < viewmatrix.numel(); i++)
    {
      std::cout << "viewmatrix[" << i << "]: " << viewmatrix.contiguous().data_ptr<float>()[i] << std::endl;
    }

    dump_tensor(dump_file, projmatrix);
    std::cout << "projmatrix shape: " << projmatrix.sizes() << std::endl;
    std::cout << "projmatrix stride: " << projmatrix.strides() << std::endl;
    std::cout << "projmatrix is_contiguous: " << projmatrix.is_contiguous() << std::endl;
    for (int i = 0; i < projmatrix.numel(); i++)
    {
      std::cout << "projmatrix[" << i << "]: " << projmatrix.contiguous().data_ptr<float>()[i] << std::endl;
    }
    std::cout << "campos shape: " << campos.sizes() << std::endl;
    std::cout << "campos stride: " << campos.strides() << std::endl;
    std::cout << "campos is_contiguous: " << campos.is_contiguous() << std::endl;

    dump_tensor(dump_file, campos);
    for (int i = 0; i < campos.numel(); i++)
    {
      std::cout << "campos[" << i << "]: " << campos.contiguous().data_ptr<float>()[i] << std::endl;
    }
    dump_file.close();

    // 执行原始函数
    rendered = cpu_rasterizer_forward(
        // geomFunc,
        // binningFunc,
        // imgFunc,
        P, degree, M,
        background.contiguous().data_ptr<float>(),
        W, H,
        means3D.contiguous().data_ptr<float>(),
        sh.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cov3D_precomp.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx,
        tan_fovy,
        prefiltered,
        out_color.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        debug);

    std::cout << "rendered: " << rendered << std::endl;
    for (int i = 0; i < 10; i++)
    {
      std::cout << "out_color[" << i << "]: " << out_color.contiguous().data_ptr<float>()[i] << std::endl;
    }
    for (int i = 0; i < 10; i++)
    {
      std::cout << "radii[" << i << "]: " << radii.contiguous().data_ptr<int>()[i] << std::endl;
    }
    // 导出计算结果
    std::fstream result_file("diff-gaussian-rasterization/data_dump/rasterizer_out.bin", std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);

    // 写入 rendered 值
    result_file.write(reinterpret_cast<const char *>(&rendered), sizeof(int));
    if (!result_file)
    {
      throw std::runtime_error("Failed to write rendered value to file");
    }

    // 写入 out_color
    dump_tensor(result_file, out_color);

    // 确保所有数据被写入
    result_file.flush();

    // // 获取当前写入位置
    // std::streampos cur_pos = result_file.tellp();
    // std::streampos out_color_pos = cur_pos - static_cast<std::streamoff>(out_color.numel() * sizeof(float));

    // // 读取 out_color 数据
    // torch::Tensor read_out_color = torch::empty_like(out_color);
    // result_file.seekg(out_color_pos);
    // result_file.read(reinterpret_cast<char *>(read_out_color.data_ptr()), out_color.numel() * sizeof(float));
    // if (!result_file)
    // {
    //   throw std::runtime_error("Failed to read out_color from file");
    // }

    // // 打印读取的数据
    // for (int i = 0; i < 10; i++)
    // {
    //   std::cout << "read_out_color[" << i << "]: " << read_out_color.data_ptr<float>()[i] << std::endl;
    // }

    // 写入 radii
    dump_tensor(result_file, radii);

    // 关闭文件
    result_file.close();
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCPU(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &radii,
    const torch::Tensor &colors,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor &dL_dout_color,
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const torch::Tensor &geomBuffer,
    const int R,
    const torch::Tensor &binningBuffer,
    const torch::Tensor &imageBuffer,
    const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if (sh.size(0) != 0)
  {
    M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  /*
  if(P != 0)
  {
    cpu_rasterizer_backward(P, degree, M, R,
    background.contiguous().data<float>(),
    W, H,
    means3D.contiguous().data<float>(),
    sh.contiguous().data<float>(),
    colors.contiguous().data<float>(),
    scales.data_ptr<float>(),
    scale_modifier,
    rotations.data_ptr<float>(),
    cov3D_precomp.contiguous().data<float>(),
    viewmatrix.contiguous().data<float>(),
    projmatrix.contiguous().data<float>(),
    campos.contiguous().data<float>(),
    tan_fovx,
    tan_fovy,
    radii.contiguous().data<int>(),
    reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
    reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
    reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
    dL_dout_color.contiguous().data<float>(),
    dL_dmeans2D.contiguous().data<float>(),
    dL_dconic.contiguous().data<float>(),
    dL_dopacity.contiguous().data<float>(),
    dL_dcolors.contiguous().data<float>(),
    dL_dmeans3D.contiguous().data<float>(),
    dL_dcov3D.contiguous().data<float>(),
    dL_dsh.contiguous().data<float>(),
    dL_dscales.contiguous().data<float>(),
    dL_drotations.contiguous().data<float>(),
    debug);
  }
  */

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor &means3D,
    torch::Tensor &viewmatrix,
    torch::Tensor &projmatrix)
{
  const int P = means3D.size(0);

  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kInt));

  if (P != 0)
  {
    // CpuRasterizer::Rasterizer::markVisible(P,
    cpu_rasterizer_markVisible(P,
                               means3D.contiguous().data_ptr<float>(),
                               viewmatrix.contiguous().data_ptr<float>(),
                               projmatrix.contiguous().data_ptr<float>(),
                               present.contiguous().data_ptr<int>());
  }

  return present;
}