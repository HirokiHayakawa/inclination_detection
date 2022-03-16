#define _USE_MATH_DEFINES
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <omp.h>
#include"cuda_func.cuh"

#ifdef _DEBUG
#pragma comment(lib, "opencv_world452d.lib")
#else
#pragma comment(lib, "opencv_world452.lib")
#endif


float __inclination_detection(uint8_t* data,
    int32_t src_h, int32_t src_w,
    int32_t out_h, int32_t out_w,
    float t1, float t2, int32_t split
) {
    int32_t x;
    float *score_arr;
    float** horizontal_sum;
    horizontal_sum = (float**)malloc(sizeof(float*) * (split + 1));
    horizontal_sum[0] = (float*)malloc(sizeof(float) * out_h * (split + 1));
    for (x = 1; x < (split + 1); x++) {
        horizontal_sum[x] = horizontal_sum[x - 1] + out_h;
    }
    score_arr = (float*)malloc(sizeof(float) * (split + 1));
#pragma omp parallel for schedule(dynamic, 1)
    for (x = 0; x <= split; x++) {
        float c, s;
        int32_t i;
        float t = t1 + (t2 - t1) * x / split;
        c = cos(t);
        s = sin(t);
        float const_y = -((float)out_w * 0.5) * s - ((float)out_h * 0.5 * c) + ((float)src_h * 0.5);
        float const_x = -((float)out_w * 0.5) * c + ((float)out_h * 0.5 * s) + ((float)src_w * 0.5);
        float all_sum = 0;
        float avg;
        int32_t nz_s = 0, nz_e = out_h;
        for (i = 0; i < out_h; i++) {
            float dsi = s * i, dci = c * i;
            int32_t j;
            horizontal_sum[x][i] = 0;
            for (j = 0; j < out_w; j++) {
                float dsj = s * j, dcj = c * j;
                int32_t sy = dsj + dci + const_y;
                int32_t sx = dcj - dsi + const_x;
                if (sx >= 0 && sx < src_w && sy >= 0 && sy < src_h) {
                    int32_t d_y_pad = src_w * sy;
                    horizontal_sum[x][i] += (255 - data[d_y_pad + sx]);
                }
            }
        }
        for (i = 0; i < out_h; i++) {
            all_sum += horizontal_sum[x][i];
        }
        for (i = 0; i < out_h; i++) {
            if (horizontal_sum[x][i]) {
                nz_s = i;
                break;
            }
        }
        for (i = out_h - 1; i >= 0; i--) {
            if (horizontal_sum[x][i]) {
                nz_e = i;
                break;
            }
        }
        avg = all_sum / (float)(nz_e - nz_s);
        float score = 0;
        for (i = nz_s; i <= nz_e; i++) {
            score += (avg - horizontal_sum[x][i]) * (avg - horizontal_sum[x][i]);
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
    free(horizontal_sum[0]);
    free(horizontal_sum);
    free(score_arr);
    return max_t;
}

float inclination_detection(cv::Mat* src) {
    uint8_t* data;
    int32_t w = src->cols;
    int32_t h = src->rows;
    int32_t i, j;
    int32_t out_h, out_w;
    // 回転後に必要なサイズ計算
    out_w = out_h = (int)ceil(sqrt((w * w) + (h * h)));

    // グレースケール化
    data = (uint8_t*)malloc(w * h);
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
    // 傾き検出
    float t = __inclination_detection(data, h, w, out_h, out_w, -M_PI / 2, M_PI / 2, 360);
    float tt = __inclination_detection(data, h, w, out_h, out_w, t - M_PI / 360, t + M_PI / 360, 100);
    free(data);
    return tt;
}

typedef struct __dot__ {
    int32_t x;
    int32_t y;
} dot;


float __inclination_detection_2(dot* dots, int32_t dot_ct,
    int32_t src_h, int32_t src_w,
    int32_t out_h, int32_t out_w,
    float t1, float t2, int32_t split
) {
    int32_t x;
    float* score_arr;
    float** horizontal_sum;
    horizontal_sum = (float**)malloc(sizeof(float*) * (split + 1));
    horizontal_sum[0] = (float*)malloc(sizeof(float) * out_h * (split + 1));
    for (x = 1; x < (split + 1); x++) {
        horizontal_sum[x] = horizontal_sum[x - 1] + out_h;
    }
    score_arr = (float*)malloc(sizeof(float) * (split + 1));

    int32_t ox, oy, ooy;
    ox = src_w / 2;
    oy = src_h / 2;
    ooy = out_h / 2;

    for (x = 0; x <= split; x++) {
        for (int i = 0; i < out_h; i++) {
            horizontal_sum[x][i] = 0;
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (x = 0; x <= split; x++) {
        float c, s;
        int32_t i;
        float t = t1 + (t2 - t1) * x / split;
        c = cos(t);
        s = sin(t);
        for (i = 0; i < dot_ct; i++) {
            int32_t y = s * (dots[i].x - ox) + c * (dots[i].y - oy) + ooy;
            horizontal_sum[x][y] += 1;
        }
        float all_sum = 0;
        float avg;
        int32_t nz_s = 0, nz_e = out_h;
        for (i = 0; i < out_h; i++) {
            all_sum += horizontal_sum[x][i];
        }
        for (i = 0; i < out_h; i++) {
            if (horizontal_sum[x][i]) {
                nz_s = i;
                break;
            }
        }
        for (i = out_h - 1; i >= 0; i--) {
            if (horizontal_sum[x][i]) {
                nz_e = i;
                break;
            }
        }
        avg = all_sum / (float)(nz_e - nz_s);
        float score = 0;
        for (i = nz_s; i <= nz_e; i++) {
            score += (avg - horizontal_sum[x][i]) * (avg - horizontal_sum[x][i]);
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

    free(horizontal_sum[0]);
    free(horizontal_sum);
    free(score_arr);
    return max_t;
}

float inclination_detection_2(cv::Mat* src) {
    uint8_t* data;
    int32_t w = src->cols;
    int32_t h = src->rows;
    int32_t i, j;
    int32_t out_h, out_w;
    int64_t dot_ct = 0;
    dot* dots;
    // 回転後に必要なサイズ計算
    out_w = out_h = (int)ceil(sqrt((w * w) + (h * h)));

    // 2値化
    data = (uint8_t*)malloc(w * h);
    for (i = 0; i < h; i++) {
        int32_t d_y_pad = w * i;
        cv::Vec3b* ptr = src->ptr<cv::Vec3b>(i);
        for (j = 0; j < w; j++) {
            cv::Vec3b bgr = ptr[j];
            uint8_t v = (uint8_t)(0.2126 * bgr[2] +
                0.7152 * bgr[1] +
                0.0722 * bgr[0]);
            data[d_y_pad + j] = 0;
            if ((255 - v) > 110) {
                data[d_y_pad + j] = 1;
                dot_ct++;
            }
        }
    }
    dots = (dot*)malloc(sizeof(dot) * dot_ct);
    dot_ct = 0;
    for (i = 0; i < h; i++) {
        int32_t d_y_pad = w * i;
        for (j = 0; j < w; j++) {
            if (data[d_y_pad + j]) {
                dots[dot_ct].x = j;
                dots[dot_ct].y = i;
                dot_ct++;
            }
        }
    }
    // 傾き検出
    float t = __inclination_detection_2(dots, dot_ct, h, w, out_h, out_w, -M_PI / 2, M_PI / 2, 360);
    float tt = __inclination_detection_2(dots, dot_ct, h, w, out_h, out_w, t - M_PI / 360, t + M_PI / 360, 100);
    free(data);
    free(dots);
    return -tt;
}

void rotate_img(float theta, cv::Mat* img) {
    float width = img->cols;
    float height = img->rows;
    cv::Point2f center = cv::Point2f((width / 2), (height / 2));//図形の中心
    float degree = theta * (180.0 / M_PI);  // 回転角度
    int32_t out_w, out_h; // 出力サイズ
    out_w = ceil(height * abs(sin(theta)) + width * abs(cos(theta)));
    out_h = ceil(height * abs(cos(theta)) + width * abs(sin(theta)));
    cv::Size size = cv::Size(out_w, out_h);
    cv::Mat change = cv::getRotationMatrix2D(center, degree, 1.0); //回転
    cv::Mat add = (cv::Mat_<float>(2, 3) << 0, 0, -width / 2 + out_w / 2, 0, 0, -height / 2 + out_h / 2); //平行移動
    change += add;
    cv::warpAffine(*img, *img, change, size, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255)); //画像の変換(アフィン変換)
}

int main()
{
    cv::Mat image;
    image = cv::imread("img001.bmp");
    if (image.empty() == true) {
        // 画像データが読み込めなかったときは終了する
        return 0;
    }
    double t1, t2, time;
    t1 = omp_get_wtime();
    //float t = inclination_detection_cuda(&image);
    //float t = inclination_detection(&image);
    float t = inclination_detection_2(&image);
    t2 = omp_get_wtime();
    time = t2 - t1;
    printf("t = %f, time = %f sec\n", t, time);

    rotate_img(t, &image);
    cv::imwrite("out.png", image);
}
