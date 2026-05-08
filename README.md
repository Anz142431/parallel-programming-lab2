# parallel-programming-lab2

并行编程实验 – 高斯消去 SIMD 加速

## 内容

- NEON 指令集 4 路向量化加速（最终版本）
- 性能对比：串行 baseline vs SIMD 优化（数据详见实验报告）

## 编译运行

```bash
# 编译 SIMD 版本
g++ -O3 -march=native main.cc -o main

# 提交到集群运行
bash test.sh 1 1
