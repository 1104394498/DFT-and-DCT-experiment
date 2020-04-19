import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def DCT_basic_experiment():
    pic_list = os.listdir('pics')
    os.makedirs(os.path.join('result', 'DCT', 'baseline'), exist_ok=True)
    os.makedirs(os.path.join('result', 'DCT', 'iDCT'), exist_ok=True)
    for i, pic_name in enumerate(pic_list):
        if pic_name[-3:] != 'jpg' and pic_name[-4:] != 'jpeg':
            continue
        pic_path = os.path.join('pics', pic_name)
        img = cv2.imread(pic_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # cv2.namedWindow(f'image_{i}')
        # cv2.imshow(f'image_{i}', img)
        # cv2.waitKey(0)

        # cv2.destroyAllWindows()
        img = np.float32(img)
        # print(img.shape)
        dct_img = cv2.dct(img)
        cv2.imwrite(os.path.join('result', 'DCT','baseline', pic_name), dct_img)

        idct_img = cv2.idct(dct_img)
        cv2.imwrite(os.path.join('result', 'DCT', 'iDCT', pic_name), idct_img)


if __name__ == '__main__':
    DCT_basic_experiment()
