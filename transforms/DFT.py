from PIL import Image
import numpy as np
import os


def DFT_per_pixel(u: int, v: int, matrix: np.ndarray):
    result = 0
    M, N = matrix.shape[0], matrix.shape[1]
    for x in range(M):
        for y in range(N):
            result += (matrix[x, y] * np.exp(-2j * np.pi * (u * x / M + v * y / N)))
    return result


def DFT(img: Image, mode='gray'):
    """
    :param img: the image to process
    :param mode: if mode = 'gray', first convert img to gray pic, and then do DFT;
                 if mode = 'RGB', first convert img to RGB pic, and then do DFT.
    :return: the image after DFT procession
    """
    if mode == 'gray':
        img = img.convert('L')
    elif mode == 'RGB':
        img = img.convert('RGB')
    else:
        raise ValueError(f'invalid mode: {mode}')

    matrix = np.array(img)

    result = np.zeros(matrix.shape, dtype=np.complex)
    M, N = matrix.shape[0], matrix.shape[1]
    if mode == 'RGB':
        for channel in matrix.shape[2]:
            for u in range(M):
                for v in range(N):
                    result[u, v, channel] = DFT_per_pixel(u, v, matrix[:, :, channel])
    else:
        for u in range(M):
            for v in range(N):
                result[u, v] = DFT_per_pixel(u, v, matrix)
    return result


if __name__ == '__main__':
    pic_list = os.listdir('../pics')
    for pic_name in pic_list:
        pic_path = os.path.join('..', 'pics', pic_name)
        img = Image.open(pic_path)
        result = DFT(img)
        print(result)
        np.fft.fft2()
