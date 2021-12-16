#define _USE_MATH_DEFINES
#include"cuda_func.cuh"
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <cmath>

#define BLOCK_DIM (32)


static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s file at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

/* CUDAカーネル関数 */
__global__
void inclination_detection_karnel(
    uint8_t* device_data,
    float* device_horizontal_sum,
    int32_t src_h, int32_t src_w,
    int32_t out_h, int32_t out_w,
    float t1, float t2, int32_t split
) {
    float c, s;
    int32_t i;
    int32_t x = blockIdx.x;
    float t = t1 + (t2 - t1) * x / split;
    c = cos(t);
    s = sin(t);
    float const_y = -((float)out_w * 0.5) * s - ((float)out_h * 0.5 * c) + ((float)src_h * 0.5);
    float const_x = -((float)out_w * 0.5) * c + ((float)out_h * 0.5 * s) + ((float)src_w * 0.5);
    i = blockIdx.y * BLOCK_DIM + threadIdx.x;
    float dsi = s * i, dci = c * i;
    int32_t j;
    int32_t dst = x * out_h + i;
    device_horizontal_sum[dst] = 0;
    for (j = 0; j < out_w; j++) {
        float dsj = s * j, dcj = c * j;
        int32_t sy = dsj + dci + const_y;
        int32_t sx = dcj - dsi + const_x;
        if (sx >= 0 && sx < src_w && sy >= 0 && sy < src_h) {
            int32_t d_y_pad = src_w * sy;
            device_horizontal_sum[dst] += (255 - device_data[d_y_pad + sx]);
        }
    }
}

float __inclination_detection_cuda(uint8_t* device_data,
    int32_t src_h, int32_t src_w,
    int32_t out_h, int32_t out_w,
    float t1, float t2, int32_t split
) {
    int32_t x;
    float* score_arr;
    float *horizontal_sum, *device_horizontal_sum;

    out_h = (out_h + BLOCK_DIM - 1) / BLOCK_DIM * BLOCK_DIM;

    horizontal_sum = (float*)malloc(sizeof(float) * out_h * (split + 1));
    HANDLE_ERROR(cudaMalloc((void**)&device_horizontal_sum, sizeof(float) * out_h * (split + 1)));

    dim3 grid(split, out_h / BLOCK_DIM);
    inclination_detection_karnel <<< grid, BLOCK_DIM >>> (
        device_data,
        device_horizontal_sum,
        src_h, src_w,
        out_h, out_w,
        t1, t2, split
    );

    HANDLE_ERROR(cudaMemcpy(horizontal_sum, device_horizontal_sum, sizeof(float) * out_h * (split + 1), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(device_horizontal_sum));

    score_arr = (float*)malloc(sizeof(float) * (split + 1));
    for (x = 0; x <= split; x++) {
        int32_t i;
        float all_sum = 0;
        int32_t nz_s = 0, nz_e = out_h;
        float avg;
        for (i = 0; i < out_h; i++) {
            all_sum += horizontal_sum[x * out_h + i];
        }
        for (i = 0; i < out_h; i++) {
            if (horizontal_sum[x * out_h + i]) {
                nz_s = i;
                break;
            }
        }
        for (i = out_h - 1; i >= 0; i--) {
            if (horizontal_sum[x * out_h + i]) {
                nz_e = i;
                break;
            }
        }
        avg = all_sum / (float)(nz_e - nz_s);
        float score = 0;
        for (i = nz_s; i <= nz_e; i++) {
            score += (avg - horizontal_sum[x * out_h + i]) * (avg - horizontal_sum[x * out_h + i]);
        }
        score /= (float)(nz_e - nz_s);
        score_arr[x] = score;
    }
    float max_score = 0;
    float max_t = 0;
    for (x = 0; x <= split; x++) {
        float t = t1 + (t2 - t1) * x / split;
        if (max_score < score_arr[x]) {
            max_score = score_arr[x];
            max_t = t;
        }
    }
    free(horizontal_sum);
    free(score_arr);
    return max_t;
}


float inclination_detection_cuda(cv::Mat* src) {
    uint8_t *data, *device_data;
    int32_t w = src->cols;
    int32_t h = src->rows;
    int32_t i, j;
    int32_t out_h, out_w;
    // 回転後に必要なサイズ計算
    out_w = out_h = (int)ceil(sqrt((w * w) + (h * h)));

    // グレースケール化
    data = (uint8_t*)malloc(w * h);
    HANDLE_ERROR(cudaMalloc((void**)&device_data, w * h));
    for (i = 0; i < h; i++) {
        int32_t d_y_pad = w * i;
        cv::Vec3b* ptr = src->ptr<cv::Vec3b>(i);
        for (j = 0; j < w; j++) {
            cv::Vec3b bgr = ptr[j];
            uint8_t v = (uint8_t)(0.2126 * bgr[2] +
                0.7152 * bgr[1] +
                0.0722 * bgr[0]);
            data[d_y_pad + j] = v;
        }
    }
    HANDLE_ERROR(cudaMemcpy(device_data, data, w * h, cudaMemcpyHostToDevice));
    // 傾き検出
    float t = __inclination_detection_cuda(device_data, h, w, out_h, out_w, -M_PI / 2, M_PI / 2, 360);
    float tt = __inclination_detection_cuda(device_data, h, w, out_h, out_w, t - M_PI / 360, t + M_PI / 360, 100);
    free(data);
    HANDLE_ERROR(cudaFree(device_data));
    return tt;
}