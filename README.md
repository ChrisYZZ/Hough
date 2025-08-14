Circular Object Notch Detection
项目概述
这是一个用于检测圆形物体（如晶圆或圆盘）中心位置和缺口（notch）旋转角度的Python项目。代码基于OpenCV库，实现图像预处理、圆检测、边缘检测和缺口角度计算。重点强调自动化和鲁棒性处理，包括噪声去除、对比度增强、自适应参数调整，以应对阴影、皱褶、反射等实际图像问题。
项目目标：

自动化：提供一个可脚本化的入口，用户只需指定输入路径，即可批量处理图像数据集。
鲁棒性：通过中值滤波、高斯模糊、CLAHE对比度增强、Canny边缘检测和极坐标采样等技术，提高对噪声和不完美图像的容忍度。如果未检测到圆或缺口，会返回None以避免崩溃。
应用场景：适用于工业检测、质量控制或Kaggle数据集如晶圆图像分析。

安装要求

Python 3.8+（测试环境：Python 3.12.3）
依赖库：

OpenCV（cv2）：用于图像处理和圆检测。
NumPy：用于数值计算。
Math：用于角度转换。
OS：用于文件路径处理。



安装依赖：
textpip install opencv-python numpy
注意：无需额外安装其他库，代码已优化为轻量级。
使用方法
1. Python脚本用法
主脚本文件：detect_notch.py

函数级使用：导入detect_center_and_angle函数，传入单个图像路径。
示例：
pythonfrom detect_notch import detect_center_and_angle

center, angle = detect_center_and_angle('path/to/image.jpg')
print(f"Center: {center}")
print(f"Angle: {angle} degrees" if angle is not None else "Not detected")

批量处理：脚本支持命令行参数指定数据集路径。
运行：
textpython detect_notch.py --dataset_path /path/to/your/dataset

--dataset_path：必选参数，指定包含.jpg或.png图像的文件夹路径。
输出：控制台打印每张图像的中心坐标和旋转角度。如果未检测到，返回"Not detected"。



2. Bash自动化脚本用法
为实现全自动化，提供一个Bash脚本run_detection.sh，用户只需编辑脚本中的路径变量，然后执行即可自动运行Python脚本。
脚本内容（run_detection.sh）：
bash#!/bin/bash

# 用户定义路径
DATASET_PATH="/path/to/your/dataset"  # 修改为你的数据集路径

# 运行Python脚本
python detect_notch.py --dataset_path "$DATASET_PATH"
使用步骤：

创建run_detection.sh文件，复制以上内容。
编辑DATASET_PATH为实际路径。
赋予执行权限：chmod +x run_detection.sh
运行：./run_detection.sh
输出：与Python脚本相同，打印处理结果。

此Bash脚本确保用户无需手动输入参数，实现一键自动化执行。适用于CI/CD管道或批量任务。
代码结构

detect_center_and_angle(image_path)：核心函数，处理单张图像。

输入：图像路径（str）。
输出：中心坐标（tuple，如( x, y )）和旋转角度（float，单位：度），或None如果未检测到。


主脚本：使用argparse解析路径，遍历文件夹处理图像。

局限性与改进建议

局限性：依赖Hough圆检测，可能在极度变形或多圆图像中失效。缺口检测假设缺口为最大连续无边缘区域。
改进：可集成机器学习模型（如YOLO）辅助圆检测；添加多线程加速批量处理。
测试：在Kaggle数据集上验证，鲁棒性高，但建议用户在自定义数据集上微调参数（如Hough的minRadius）。
