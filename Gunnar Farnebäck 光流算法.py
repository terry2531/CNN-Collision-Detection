import cv2
import numpy as np
import glob

def gf(n):
    # 读取所有图片，按文件名排序
    image_files = sorted(glob.glob(r'C:\Users\A\PycharmProjects\PythonProject\project 002\video\001\*.png'))  # 修改为你的路径

    if len(image_files) < 2:
        print("Error: At least two images are required.")
        exit()

    # 读取第一帧
    frame1_gray = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    frame1_bgr = cv2.imread(image_files[0])  # 读取彩色图像

    if frame1_gray is None or frame1_bgr is None:
        print("Error: Unable to load the first image.")
        exit()

    h, w = frame1_gray.shape  # 获取图像尺寸
    total_size = len(image_files) * h * w  # 所有帧图像的总面积
    size = 0  # 框选区域的总面积

    for i in range(1, len(image_files)):
        frame2_gray = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
        frame2_bgr = cv2.imread(image_files[i])  # 读取彩色图像

        if frame2_gray is None or frame2_bgr is None:
            print(f"Error: Unable to load image {image_files[i]}")
            continue

        # 确保两张图大小一致
        if frame1_gray.shape != frame2_gray.shape:
            frame2_gray = cv2.resize(frame2_gray, (w, h))
            frame2_bgr = cv2.resize(frame2_bgr, (w, h))

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 计算光流的幅度和方向
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # **筛选显著运动区域**
        motion_mask = magnitude > 0.5  # 设定一个幅度阈值来区分显著运动
        motion_mask = motion_mask.astype(np.uint8) * 255  # 转换为二值图像（0或255）

        # **去除噪声**
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # **查找轮廓**
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # **在显著运动区域上绘制绿色框并计算面积**
        total_motion_area = 0  # 当前帧的总运动区域面积
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # 排除较小的噪声，设定阈值
                x, y, w, h = cv2.boundingRect(contour)
                motion_area = w * h  # 当前框选区域的面积
                total_motion_area += motion_area
                # 在显著运动区域上绘制绿色框（如果需要）
                cv2.rectangle(frame2_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色框

        # 结果
        cv2.imshow("Optical Flow with Bounding Box", frame2_bgr)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # 更新前一帧
        frame1_gray = frame2_gray.copy()
        frame1_bgr = frame2_bgr.copy()

        # 累加所有帧的运动区域面积
        size += total_motion_area

    # 计算运动区域占总区域的百分比
    motion_percentage = (size / total_size) * 100

    # 最后输出占比
    print(f"Total Motion Area: {size} pixels")
    print(f"Total Area: {total_size} pixels")
    print(f"Motion Area Percentage: {motion_percentage:.2f}%")


gf(0)