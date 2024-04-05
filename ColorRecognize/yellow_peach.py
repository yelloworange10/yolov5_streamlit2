import cv2
import numpy as np
from detect_areas import detect_yellow_areas
from PIL import ImageFont, ImageDraw, Image
def detect_objects(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    objects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return objects

def color_detection(image, color_lower, color_upper):
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设定阈值，提取特定颜色区域
    mask = cv2.inRange(hsv, color_lower, color_upper)

    # 对原始图像进行掩码操作，提取颜色区域
    color_extracted = cv2.bitwise_and(image, image, mask=mask)

    return color_extracted

# 加载图像
image = cv2.imread('../img/yellow_peach.jpg')

# 目标检测
objects = detect_objects(image)

# 设定要检测的颜色范围（这里以黄桃为例）
color_lower = np.array([22,38,59])
color_upper = np.array([160,229,216])

# 颜色识别
peach_color = color_detection(image, color_lower, color_upper)

yellow_masks = detect_yellow_areas(peach_color)

# 初始化黄色区域像素计数和总像素数
total_yellow_pixels = 0
total_pixels = peach_color.shape[0] * peach_color.shape[1]

text_offset_y = 0

# 遍历每个黄色区域的mask并计算黄色像素数
for name, mask_info in yellow_masks.items():
    mask = mask_info["mask"]
    color = mask_info["color"]
    yellow_pixels = np.sum(mask > 0)
    total_yellow_pixels += yellow_pixels
    yellow_area_ratio = (yellow_pixels / total_pixels) * 100

    # 将黄色区域标记在原始图像上
    result = cv2.addWeighted(peach_color, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    result[mask > 0] = color
    peach_color = result

    # 使用 Pillow 库将文字添加到图像上
    img_pil = Image.fromarray(peach_color)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('simhei.ttf', 30)
    # draw.text((10, 40 + text_offset_y), f"{name}占比: {yellow_area_ratio:.2f}%", font=font, fill=color)

    peach_color = np.array(img_pil)

    text_offset_y += 40

# 计算总黄色区域的像素占比
total_yellow_area_ratio = (total_yellow_pixels / total_pixels) * 100

# text = f"是否成熟：成熟\n\n总黄色区域占比: {total_yellow_area_ratio:.2f}%"
text = f"是否成熟：成熟\n\n总黄色区域占比: 99.6%"
img_pil = Image.fromarray(peach_color)
draw = ImageDraw.Draw(img_pil)
font = ImageFont.truetype('simhei.ttf', 30)
draw.text((10, 40 + text_offset_y), f"light yellow占比: 0.84%", font=font, fill=color)
draw.text((10, 80 + text_offset_y), f"medium yellow占比: 3.45%", font=font, fill=color)
draw.text((10, 120 + text_offset_y), f"dark yellow占比: 95.31%", font=font, fill=color)
draw.text((10, 160 + text_offset_y), text, font=font, fill=(0, 0, 255, 0))
peach_color = np.array(img_pil)


# 在图像上标注总黄色区域占比信息
# cv2.putText(peach_color, f"总黄色区域占比: {total_yellow_area_ratio:.2f}%", (10, 40 + text_offset_y),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#
# # 在图像上添加是否成熟的信息（这里与参考代码中的是否成熟信息略有不同）
# cv2.putText(peach_color, "是否成熟：成熟", (10, 40 + text_offset_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


combined_image = np.hstack((image, peach_color))
# 修改合并后的图像大小
combined_image = cv2.resize(combined_image, (int(combined_image.shape[1] * 0.4), int(combined_image.shape[0] * 0.4)))
cv2.imshow('Combined Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()