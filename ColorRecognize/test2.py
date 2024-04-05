import cv2

# 读取桃子图像
image = cv2.imread('../img/test3.jpg')

# 将图像转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用 Canny 边缘检测器进行边缘检测
edges = cv2.Canny(gray, 100, 200)  # 100 和 200 是 Canny 边缘检测器的低阈值和高阈值参数

# 显示边缘检测结果
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
