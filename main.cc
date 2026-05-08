#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
// 自行添加需要的头文件
#include <cmath>
#include <cassert>
#include <random>
#include <arm_neon.h>   // NEON 指令集

// 高斯消去（NEON SIMD 加速版本）
void gaussianElimination(std::vector<std::vector<float>>& A) {
    int n = (int)A.size();
    
    // 把矩阵转成一维数组，连续内存方便 SIMD 加载
    std::vector<float> A_flat(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_flat[i * n + j] = A[i][j];
        }
    }
    
    for (int k = 0; k < n; ++k) {
        float pivot = A_flat[k * n + k];
        if (std::abs(pivot) < 1e-8f) {
            std::cerr << "错误：pivot 太小 (k=" << k << ")" << std::endl;
            return;
        }
        
        // ---------- 1) 归一化第k行（一次4个float）----------
        float32x4_t vpivot = vdupq_n_f32(pivot);
        int j = k + 1;
        for (; j + 4 <= n; j += 4) {
            float32x4_t va = vld1q_f32(&A_flat[k * n + j]);
            va = vdivq_f32(va, vpivot);
            vst1q_f32(&A_flat[k * n + j], va);
        }
        // 尾部不足4个的处理
        for (; j < n; ++j) {
            A_flat[k * n + j] /= pivot;
        }
        A_flat[k * n + k] = 1.0f;
        
        // ---------- 2) 消去下面所有行（一次4个float）----------
        for (int i = k + 1; i < n; ++i) {
            float factor = A_flat[i * n + k];
            if (std::abs(factor) < 1e-8f) continue;
            
            float32x4_t vfactor = vdupq_n_f32(factor);
            j = k + 1;
            for (; j + 4 <= n; j += 4) {
                float32x4_t vrow_k = vld1q_f32(&A_flat[k * n + j]);
                float32x4_t vrow_i = vld1q_f32(&A_flat[i * n + j]);
                vrow_i = vsubq_f32(vrow_i, vmulq_f32(vfactor, vrow_k));
                vst1q_f32(&A_flat[i * n + j], vrow_i);
            }
            // 尾部不足4个的处理
            for (; j < n; ++j) {
                A_flat[i * n + j] -= factor * A_flat[k * n + j];
            }
            A_flat[i * n + k] = 0.0f;
        }
    }
    
    // 把结果写回原矩阵（保持接口不变）
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = A_flat[i * n + j];
        }
    }
}

// 生成随机矩阵（可逆，避免除零）
std::vector<std::vector<float>> generateMatrix(int n) {
    std::vector<std::vector<float>> A(n, std::vector<float>(n));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = dis(gen);
        }
    }
    return A;
}

// 打印矩阵（仅用于小规模测试）
void printMatrix(const std::vector<std::vector<float>>& A) {
    int n = (int)A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(10) << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[])
{
    auto Start = std::chrono::high_resolution_clock::now();
    
    // ========== 正确性测试（默认注释掉，需要时取消注释）==========
    // const int test_n = 4;
    // std::cout << "===== 正确性测试: n = " << test_n << " =====" << std::endl;
    // auto A_test = generateMatrix(test_n);
    // std::cout << "原始矩阵:" << std::endl;
    // printMatrix(A_test);
    // gaussianElimination(A_test);
    // std::cout << "\n高斯消去后的上三角矩阵:" << std::endl;
    // printMatrix(A_test);
    // std::cout << "请检查对角线以下是否全为0（理想情况下应为0）\n" << std::endl;
    
    // ========== 性能测试 ==========
    std::vector<int> sizes = {64, 128, 256, 512, 1024};
    std::cout << "===== 性能测试 =====" << std::endl;
    for (int n : sizes) {
        auto A = generateMatrix(n);
        auto start = std::chrono::high_resolution_clock::now();
        gaussianElimination(A);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "n = " << n << " time: " << elapsed_ms << " ms" << std::endl;
    }
    
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::ratio<1,1000>> elapsed = End - Start;
    std::cout << "total time: " << elapsed.count() << " ms" << std::endl;
    return 0;
}