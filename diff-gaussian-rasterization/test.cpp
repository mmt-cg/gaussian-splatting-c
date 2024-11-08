#include <fstream>
#include <vector>
#include <cstring>
#include <iostream>
#include "cpu_rasterizer/rasterizer.h"
#include <cmath>

void compare_vectors(const std::string &name,
                     const std::vector<float> &expected,
                     const std::vector<float> &actual,
                     std::ofstream &log_file)
{
    const float epsilon = 1e-6;

    if (expected.size() != actual.size())
    {
        log_file << "Size mismatch in " << name << "\n";
        return;
    }

    if (expected.size() == 1)
    {
        float difference = std::abs(expected[0] - actual[0]);
        if (difference > epsilon)
        {
            log_file << "Scalar " << name << " mismatch: Expected " << expected[0]
                     << ", Got " << actual[0]
                     << ", Difference: " << difference << "\n";
        }
    }
    else
    {
        float total_error = 0.0f;
        float max_error = 0.0f;
        for (size_t i = 0; i < expected.size(); i++)
        {
            float error = std::abs(expected[i] - actual[i]);
            total_error += error;
            if (error > max_error)
            {
                max_error = error;
            }
            if (error > epsilon)
            {
                log_file << "Index " << i << ": Expected " << expected[i]
                         << ", Got " << actual[i]
                         << ", Error: " << error << "\n";
            }
        }
        float average_error = total_error / expected.size();
        log_file << "Vector " << name << " average error: " << average_error
                 << ", maximum error: " << max_error << "\n";
    }
}

int main()
{
    std::ofstream log_file("debug/record.log");

    // 读取输入数据
    std::ifstream in_file("data_dump/rasterizer_in.bin", std::ios::binary);
    if (!in_file.is_open())
    {
        std::cerr << "Failed to open input file" << std::endl;
        return 1;
    }

    // 读取标量参数
    int P, degree, M, W, H;
    float scale_modifier, tan_fovx, tan_fovy;
    bool prefiltered, debug;

    in_file.read(reinterpret_cast<char *>(&P), sizeof(int));
    in_file.read(reinterpret_cast<char *>(&degree), sizeof(int));
    in_file.read(reinterpret_cast<char *>(&M), sizeof(int));
    in_file.read(reinterpret_cast<char *>(&W), sizeof(int));
    in_file.read(reinterpret_cast<char *>(&H), sizeof(int));
    in_file.read(reinterpret_cast<char *>(&scale_modifier), sizeof(float));
    in_file.read(reinterpret_cast<char *>(&tan_fovx), sizeof(float));
    in_file.read(reinterpret_cast<char *>(&tan_fovy), sizeof(float));
    in_file.read(reinterpret_cast<char *>(&prefiltered), sizeof(bool));
    in_file.read(reinterpret_cast<char *>(&debug), sizeof(bool));
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

    // 为每个数组分配内存并直接读取
    std::vector<float> background(3);
    std::vector<float> means3D(P * 3);
    std::vector<float> sh(P * M * 3);
    std::vector<float> colors(0);
    std::vector<float> opacity(P); // 修正：P * 1 改为 P
    std::vector<float> scales(P * 3);
    std::vector<float> rotations(P * 4);
    std::vector<float> cov3D_precomp(0);
    std::vector<float> viewmatrix(4 * 4);
    std::vector<float> projmatrix(4 * 4);
    std::vector<float> campos(3);

    // 直接读取数据
    in_file.read(reinterpret_cast<char *>(background.data()), 3 * sizeof(float));
    // for (int i = 0; i < 3; i++)
    // {
    //     std::cout << "background[" << i << "]: " << background[i] << std::endl;
    // }
    in_file.read(reinterpret_cast<char *>(means3D.data()), P * 3 * sizeof(float));
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "means3D[" << i << "]: " << means3D[i] << std::endl;
    // }
    in_file.read(reinterpret_cast<char *>(sh.data()), P * M * 3 * sizeof(float));
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "sh[" << i << "]: " << sh[i] << std::endl;
    // }
    in_file.read(reinterpret_cast<char *>(colors.data()), 0);
    in_file.read(reinterpret_cast<char *>(opacity.data()), P * sizeof(float));
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "opacity[" << i << "]: " << opacity[i] << std::endl;
    // }
    in_file.read(reinterpret_cast<char *>(scales.data()), P * 3 * sizeof(float));
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "scales[" << i << "]: " << scales[i] << std::endl;
    // }
    in_file.read(reinterpret_cast<char *>(rotations.data()), P * 4 * sizeof(float));
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "rotations[" << i << "]: " << rotations[i] << std::endl;
    // }
    in_file.read(reinterpret_cast<char *>(cov3D_precomp.data()), 0);
    in_file.read(reinterpret_cast<char *>(viewmatrix.data()), 16 * sizeof(float));
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "viewmatrix[" << i << "]: " << viewmatrix[i] << std::endl;
    // }
    in_file.read(reinterpret_cast<char *>(projmatrix.data()), 16 * sizeof(float));
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "projmatrix[" << i << "]: " << projmatrix[i] << std::endl;
    // }
    in_file.read(reinterpret_cast<char *>(campos.data()), 3 * sizeof(float));
    // for (int i = 0; i < 3; i++)
    // {
    //     std::cout << "campos[" << i << "]: " << campos[i] << std::endl;
    // }

    in_file.close();

    // 输出相关
    std::vector<float> out_color(3 * H * W);
    std::vector<int> radii(P);

    // 调用 rasterizer
    int rendered = cpu_rasterizer_forward(
        P, degree, M,
        background.data(),
        W, H,
        means3D.data(),
        sh.data(),
        colors.data(),
        opacity.data(),
        scales.data(),
        scale_modifier,
        rotations.data(),
        cov3D_precomp.data(),
        viewmatrix.data(),
        projmatrix.data(),
        campos.data(),
        tan_fovx,
        tan_fovy,
        prefiltered,
        out_color.data(),
        radii.data(),
        debug);

    // 读取实际输出
    std::ifstream out_file("data_dump/rasterizer_out.bin", std::ios::binary);
    if (!out_file.is_open())
    {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }
    // std::cout << "actual_rendered: " << rendered << std::endl;
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "out_color[" << i << "]: " << out_color[i] << std::endl;
    // }
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "radii[" << i << "]: " << radii[i] << std::endl;
    // }

    int expected_rendered;
    out_file.read(reinterpret_cast<char *>(&expected_rendered), sizeof(int));
    // std::cout << "expected_rendered: " << expected_rendered << std::endl;
    std::vector<float> expected_out_color(3 * H * W);
    out_file.read(reinterpret_cast<char *>(expected_out_color.data()), 3 * H * W * sizeof(float));
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "expected_out_color[" << i << "]: " << expected_out_color[i] << std::endl;
    // }
    std::vector<int> expected_radii(P);
    out_file.read(reinterpret_cast<char *>(expected_radii.data()), P * sizeof(int));
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "expected_radii[" << i << "]: " << expected_radii[i] << std::endl;
    // }

    out_file.close();

    // 比较结果
    if (rendered != expected_rendered)
    {
        log_file << "Rendered count mismatch: Expected " << expected_rendered
                 << ", Got " << rendered << "\n";
    }

    compare_vectors("out_color", expected_out_color, out_color, log_file);
    compare_vectors("radii",
                    std::vector<float>(expected_radii.begin(), expected_radii.end()),
                    std::vector<float>(radii.begin(), radii.end()), log_file);

    log_file.close();
    return 0;
}
