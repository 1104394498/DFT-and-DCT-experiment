from DFT import *
from DCT import *

if __name__ == '__main__':
    DFT_basic_experiment()
    Fourier_filtering(mode='lowpass', pass_ratio=0.1)
    Fourier_filtering(mode='highpass', pass_ratio=0.7)
    DCT_basic_experiment()
