import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import ImageFont, ImageDraw, Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from detect_areas import detect_red_areas

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()


def detect_peach(image, threshold=0.1):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    with torch.no_grad():
        predictions = model([img_tensor])

    boxes = predictions[0]['boxes'].numpy()
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()

    peach_boxes = boxes[(labels == 53) & (scores >= threshold)]
    return peach_boxes


image = cv2.imread('../img/color_test.jpg')
original_image = image.copy()  # 创建原图的副本以供后续使用
peach_boxes = detect_peach(image)

for box in peach_boxes:
    x1, y1, x2, y2 = box.astype(int)
    peach_roi = image[y1:y2, x1:x2]
    red_area_masks = detect_red_areas(peach_roi)

    total_pixels = peach_roi.shape[0] * peach_roi.shape[1]
    total_red_pixels = 0

    text_offset_y = 0
    for name, mask_info in red_area_masks.items():
        mask = mask_info["mask"]
        color = mask_info["color"]
        red_pixels = np.sum(mask > 0)
        total_red_pixels += red_pixels
        red_area_ratio = (red_pixels / total_pixels) * 100

        result = cv2.addWeighted(peach_roi, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        result[mask > 0] = color
        image[y1:y2, x1:x2] = result

        text = f"{name}占比: {red_area_ratio:.2f}%"
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype('simhei.ttf', 30)
        draw.text((x1, y1 - 40 - text_offset_y), text, font=font, fill=color)
        image = np.array(img_pil)

        text_offset_y += 40

    total_red_area_ratio = (total_red_pixels / total_pixels) * 100

    text = f"是否成熟：成熟\n总红色区域占比: {total_red_area_ratio:.2f}%"
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('simhei.ttf', 30)
    draw.text((x1, y1 - 60 - text_offset_y), text, font=font, fill=(0, 0, 255, 0))
    image = np.array(img_pil)

# 调整图像大小
display_size = (600, 400)  # 你可以根据需要调整这个大小
resized_image = cv2.resize(image, display_size)
resized_original_image = cv2.resize(original_image, display_size)

# 将原图和结果图并排显示
comparison_image = cv2.hconcat([resized_original_image, resized_image])

# 显示结果图像
cv2.imshow('Result', comparison_image)
cv2.waitKey(0)
cv2.destroyAllWindows()