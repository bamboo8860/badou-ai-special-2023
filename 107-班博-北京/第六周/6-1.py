import cv2
import numpy as np

# 读取两张图像
img1 = cv2.imread('iphone1.png', 0)  # 读取灰度图像
img2 = cv2.imread('iphone2.png', 0)

# 初始化 SIFT 检测器
sift = cv2.xfeatures2d.SIFT_create()

# 寻找关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher()

# 使用 KNN 匹配
matches = bf.knnMatch(des1, des2, k=2)

# 应用比率测试
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# 绘制匹配结果
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# 显示匹配结果
cv2.imshow('Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

