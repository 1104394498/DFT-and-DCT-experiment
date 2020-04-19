import numpy as np
import os
from PIL import Image, ImageChops
from matplotlib import pyplot as plt


def DFT2(img: np.ndarray) -> np.ndarray:
    ret = np.fft.fft2(img)
    ret = np.fft.fftshift(ret)
    return ret


def two_dims_DFT(img: np.ndarray) -> np.ndarray:
    """
    First do DFT in the y-dimension, and then do DFT in the x-dimension
    :param img: the image to process
    :return: the result of two dims DFT
    """
    result = np.zeros(img.shape, dtype=np.complex)
    for i in range(img.shape[0]):
        result[i, :] = np.fft.fft(img[i, :])
    for j in range(img.shape[1]):
        result[:, j] = np.fft.fft(result[:, j])
    result = np.fft.fftshift(result)
    return result


def frequency_spectrum(DFT_result: np.ndarray, pic_name: str, folder: str):
    f = np.abs(DFT_result)
    f = np.log(f + 1)

    plt.imshow(f)
    plt.title(f'frequency spectrum for {pic_name}')
    frequency_spectrum_folder = os.path.join('result', 'DFT', 'frequency_spectrum', folder)
    os.makedirs(frequency_spectrum_folder, exist_ok=True)
    plt.savefig(os.path.join(frequency_spectrum_folder, pic_name))


def phase_position(DFT_result: np.ndarray, pic_name: str, folder: str):
    real = np.real(DFT_result)
    imag = np.imag(DFT_result)

    f = np.arctan(imag / real)

    plt.imshow(f)
    plt.title(f'phase position for {pic_name}')
    frequency_spectrum_folder = os.path.join('result', 'DFT', 'phase_position', folder)
    os.makedirs(frequency_spectrum_folder, exist_ok=True)
    plt.savefig(os.path.join(frequency_spectrum_folder, pic_name))


def ImgOffset(Img, xoff, yoff):
    width, height = Img.size
    c = ImageChops.offset(Img, xoff, yoff)
    # c.paste((0, 0, 0), (0, 0, xoff, height))
    # c.paste((0, 0, 0), (0, 0, width, yoff))
    return c


def DFT_basic_experiment():
    pic_list = os.listdir('pics')
    for pic_name in pic_list:
        if pic_name[-3:] != 'jpg' and pic_name[-4:] != 'jpeg':
            continue
        pic_path = os.path.join('pics', pic_name)
        img = Image.open(pic_path)
        img = img.convert('L')

        img_array = np.array(img)

        DFT_result = DFT2(img_array)

        """
        1. 傅立叶反变换
        """
        iDFT_result = np.fft.ifftshift(DFT_result)
        iDFT_result = np.real(np.fft.ifft2(iDFT_result))
        plt.imshow(iDFT_result, cmap='gray')
        plt.title(f'iDFT for {pic_name}')
        iDFT_path = os.path.join('result', 'DFT', 'iDFT')
        os.makedirs(iDFT_path, exist_ok=True)
        plt.savefig(os.path.join(iDFT_path, pic_name))

        """
        2. 绘制频域图和相位图
        """
        # 绘制频域图
        frequency_spectrum(DFT_result, pic_name, 'baseline')
        # 绘制相位图
        phase_position(DFT_result, pic_name, 'baseline')

        """
        3. 验证傅立叶变换的性质
        """
        # 可分离性
        two_dims_DFT_result = two_dims_DFT(img_array)
        frequency_spectrum(two_dims_DFT_result, pic_name, 'separate')
        phase_position(two_dims_DFT_result, pic_name, 'separate')

        # 平移性
        offset_img = ImgOffset(img, img_array.shape[0] // 2, img_array.shape[1] // 2)
        offset_img.show()
        offset_img_array = np.array(offset_img)
        offset_DFT_result = DFT2(offset_img_array)
        frequency_spectrum(offset_DFT_result, pic_name, 'offset')
        phase_position(offset_DFT_result, pic_name, 'offset')

        # 旋转性
        rotate_img = img.rotate(30)
        rotate_img_array = np.array(rotate_img)
        rotate_DFT_result = DFT2(rotate_img_array)
        frequency_spectrum(rotate_DFT_result, pic_name, 'rotate')
        phase_position(rotate_DFT_result, pic_name, 'rotate')

        # 尺度变换性
        ratio = 0.5
        resize_img = img.resize((int(img.size[0] * 0.5), int(img.size[1] * 0.5)))
        resize_img_array = np.array(resize_img)
        resize_DFT_result = DFT2(resize_img_array)
        frequency_spectrum(resize_DFT_result, pic_name, 'resize')
        phase_position(resize_DFT_result, pic_name, 'resize')


def Fourier_filtering(mode: str = 'lowpass', pass_ratio: float = 0.9):
    assert mode == 'lowpass' or mode == 'highpass'
    if mode == 'lowpass':
        folder_path = os.path.join('result', 'DFT', 'filtering', 'lowpass')
    else:
        folder_path = os.path.join('result', 'DFT', 'filtering', 'highpass')
    os.makedirs(folder_path, exist_ok=True)

    pic_list = os.listdir('pics')
    for pic_name in pic_list:
        if pic_name[-3:] != 'jpg' and pic_name[-4:] != 'jpeg':
            continue
        pic_path = os.path.join('pics', pic_name)
        img = Image.open(pic_path)
        img = img.convert('L')

        img_array = np.array(img)

        DFT_result = DFT2(img_array)

        height, width = DFT_result.shape
        crow, ccol = height // 2, width // 2
        if mode == 'lowpass':
            lowpass_DFT_result = DFT_result
            mask = np.zeros(lowpass_DFT_result.shape)
            mask[int(crow - pass_ratio / 2 * crow): int(crow + pass_ratio / 2 * crow),
            int(ccol - pass_ratio / 2 * ccol): int(ccol + pass_ratio / 2 * ccol)] = 1
            lowpass_DFT_result = lowpass_DFT_result * mask
            lowpass_image = np.fft.ifftshift(lowpass_DFT_result)
            lowpass_image = np.real(np.fft.ifft2(lowpass_image))
            plt.imshow(lowpass_image, cmap='gray')
            plt.title(f'lowpass filtering for {pic_name}, pass ratio = {pass_ratio}')
            plt.savefig(os.path.join(folder_path, pic_name))
        else:
            highpass_DFT_result = DFT_result
            half_mask_ratio = (1 - pass_ratio) / 2
            highpass_DFT_result[int(crow - half_mask_ratio * height): int(crow + half_mask_ratio * height),
            int(ccol - half_mask_ratio * width): int(ccol + half_mask_ratio * width)] = 0
            highpass_image = np.fft.ifftshift(highpass_DFT_result)
            highpass_image = np.real(np.fft.ifft2(highpass_image))
            plt.imshow(highpass_image, cmap='gray')
            plt.title(f'highpass filtering for {pic_name}, pass ratio = {pass_ratio}')

            plt.savefig(os.path.join(folder_path, pic_name))
