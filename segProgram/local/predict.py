import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from .src import UNet
import cv2
# import sys

def predict_img(img_path, resolution):
    classes = 1  # exclude background
    PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    weights_path = os.path.join(PROJECT_ROOT, 'local/best_model_till_91_20230415_0424.pth')
    # weights_path = '/Users/pili/PycharmProjects/buildingSeg/segProgram/local/best_model_till_91_20230415_0424.pth'
    # img_folder = "/home/pili/MyDataSet/Building_Seg/mass_build/png/test"
    # get devices
    device = torch.device("cpu")
    print("using {} device.".format(device))
    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    #roi_mask_path = "./DRIVE/test/mask/03_test_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    #assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."
    #print(img_one)
    # mean = (0.709, 0.381, 0.224)
    # std = (0.127, 0.079, 0.043)
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    # load roi mask
    #roi_img = Image.open(roi_mask_path).convert('L')
    #roi_img = np.array(roi_img)

    # load image
    # with open(img_path, 'rb') as f:
    #     original_img = Image.open(f)
    #     original_img.load()
    original_img = Image.open(img_path).convert('RGB')
    # print("original_img.size:", original_img.size)
    # original_img.save('/Users/pili/Documents/output.png')
    # from pil image to tensor and normalize
    data_transform = transforms.Compose([
        # transforms.Resize(1600),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # print('img.shape:', img.shape)
    model.eval()  # 进入验证模式
    with torch.no_grad():
        start_time = time.time()

        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        output = model(img.to(device))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        #prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        count = sum(1 for pixel in mask.getdata() if pixel != 0)
        # print("mask.size:", mask.size)
        space = count / 28.0 * resolution

        end_time = time.time()
        print("代码运行时间：", (end_time-start_time) * 1000, "ms")
        # print("count: ", count)
        img_bgr = cv2.cvtColor(np.asarray(original_img), cv2.COLOR_RGB2BGR)
        contours, im = cv2.findContours(prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
        pre_img = cv2.drawContours(img_bgr, contours=contours, contourIdx=-1, color=(64, 224, 208), thickness=1)
        image_pil = Image.fromarray(cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB))
        # image_pil.save('/Users/pili/Documents/output.png')
        return space


if __name__ == '__main__':
    predict_img()
