#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "cpu_rasterizer/rasterizer.h"
#include <math.h>

void compare_float_arrays(const char* name, float* expected, float* actual, 
                         int size, FILE* log_file) {
    const float epsilon = 1e-6f;
    if (size == 1) {
        // 标量输出，计算差值
        float difference = fabsf(expected[0] - actual[0]);
        if (difference > epsilon) {
            fprintf(log_file, "Scalar %s mismatch: Expected %f, Got %f, Difference: %f\n",
                    name, expected[0], actual[0], difference);
        }
    } else {
        // 向量输出，计算平均误差和最大误差
        float total_error = 0.0f;
        float max_error = 0.0f;
        for (int i = 0; i < size; i++) {
            float error = fabsf(expected[i] - actual[i]);
            total_error += error;
            if (error > max_error) {
                max_error = error;
            }
            if (error > epsilon) {
                fprintf(log_file, "Index %d: Expected %f, Got %f, Error: %f\n",
                        i, expected[i], actual[i], error);
            }
        }
        float average_error = total_error / size;
        fprintf(log_file, "Vector %s average error: %f, maximum error: %f\n",
                name, average_error, max_error);
    }
}

void compare_int_arrays(const char* name, int* expected, int* actual, 
                       int size, FILE* log_file) {
    const int epsilon = 0; // 对于整数，通常不需要误差范围
    if (size == 1) {
        // 标量输出，计算差值
        if (expected[0] != actual[0]) {
            fprintf(log_file, "Scalar %s mismatch: Expected %d, Got %d\n",
                    name, expected[0], actual[0]);
        }
    } else {
        // 向量输出，计算最大差值
        int max_error = 0;
        for (int i = 0; i < size; i++) {
            int error = abs(expected[i] - actual[i]);
            if (error > max_error) {
                max_error = error;
            }
            if (error > epsilon) {
                fprintf(log_file, "Index %d: Expected %d, Got %d, Error: %d\n",
                        i, expected[i], actual[i], error);
            }
        }
        fprintf(log_file, "Vector %s maximum error: %d\n",
                name, max_error);
    }
}

int main() {
    FILE* log_file = fopen("debug/record.log", "w");
    if (!log_file) {
        printf("Failed to open log file\n");
        return 1;
    }

    // 读取输入数据
    FILE* in_file = fopen("data_dump/rasterizer_in.bin", "rb");
    if (!in_file) {
        printf("Failed to open input file\n");
        return 1;
    }

    // 读取标量参数
    int P, degree, M, W, H;
    float scale_modifier, tan_fovx, tan_fovy;
    bool prefiltered, debug;

    fread(&P, sizeof(int), 1, in_file);
    fread(&degree, sizeof(int), 1, in_file);
    fread(&M, sizeof(int), 1, in_file);
    fread(&W, sizeof(int), 1, in_file);
    fread(&H, sizeof(int), 1, in_file);
    fread(&scale_modifier, sizeof(float), 1, in_file);
    fread(&tan_fovx, sizeof(float), 1, in_file);
    fread(&tan_fovy, sizeof(float), 1, in_file);
    fread(&prefiltered, sizeof(bool), 1, in_file);
    fread(&debug, sizeof(bool), 1, in_file);

    // 分配内存并读取输入tensor
    float* background = (float*)malloc(3 * sizeof(float));
    float* means3D = (float*)malloc(P * 3 * sizeof(float));
    float* sh = (float*)malloc(P * M * 3 * sizeof(float));
    float* colors = NULL;
    float* opacity = (float*)malloc(P * sizeof(float));
    float* scales = (float*)malloc(P * 3 * sizeof(float));
    float* rotations = (float*)malloc(P * 4 * sizeof(float));
    float* cov3D_precomp = NULL;
    float* viewmatrix = (float*)malloc(16 * sizeof(float));
    float* projmatrix = (float*)malloc(16 * sizeof(float));
    float* campos = (float*)malloc(3 * sizeof(float));

    fread(background, sizeof(float), 3, in_file);
    fread(means3D, sizeof(float), P * 3, in_file);
    fread(sh, sizeof(float), P * M * 3, in_file);
    // fread(colors, sizeof(float), 0, in_file);
    fread(opacity, sizeof(float), P, in_file);
    fread(scales, sizeof(float), P * 3, in_file);
    fread(rotations, sizeof(float), P * 4, in_file);
    // fread(cov3D_precomp, sizeof(float), 0, in_file);
    fread(viewmatrix, sizeof(float), 16, in_file);
    fread(projmatrix, sizeof(float), 16, in_file);
    fread(campos, sizeof(float), 3, in_file);

    // 读取预期输出
    float* expected_out_color = (float*)malloc(3 * H * W * sizeof(float));
    int* expected_radii = (int*)malloc(P * sizeof(int));

    fclose(in_file);

    // 准备实际输出缓冲区
    float* actual_out_color = (float*)malloc(3 * H * W * sizeof(float));
    int* actual_radii = (int*)malloc(P * sizeof(int));
    memset(actual_out_color, 0, 3 * H * W * sizeof(float));
    memset(actual_radii, 0, P * sizeof(int));

    // 调用 rasterizer
    int rendered = cpu_rasterizer_forward(
        P, degree, M,
        background,
        W, H,
        means3D,
        sh,
        colors,
        opacity,
        scales,
        scale_modifier,
        rotations,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        campos,
        tan_fovx,
        tan_fovy,
        prefiltered,
        actual_out_color,
        actual_radii,
        debug);

    // 读取预期的rendered值
    FILE* out_file = fopen("data_dump/rasterizer_out.bin", "rb");
    if (!out_file) {
        printf("Failed to open output file\n");
        return 1;
    }

    int expected_rendered;
    fread(&expected_rendered, sizeof(int), 1, out_file);
    fread(expected_out_color, sizeof(float), 3 * H * W, out_file);
    fread(expected_radii, sizeof(int), P, out_file);
    fclose(out_file);

    // 比较结果
    if (rendered != expected_rendered) {
        fprintf(log_file, "Rendered count mismatch: Expected %d, Got %d\n",
                expected_rendered, rendered);
    }

    compare_float_arrays("out_color", expected_out_color, actual_out_color,
                        3 * H * W, log_file);
    compare_int_arrays("radii", expected_radii, actual_radii, P, log_file);

    // 清理内存
    free(background);
    free(means3D);
    free(sh);
    free(colors);
    free(opacity);
    free(scales);
    free(rotations);
    free(cov3D_precomp);
    free(viewmatrix);
    free(projmatrix);
    free(campos);
    free(expected_out_color);
    free(expected_radii);
    free(actual_out_color);
    free(actual_radii);

    fclose(log_file);
    return 0;
} 