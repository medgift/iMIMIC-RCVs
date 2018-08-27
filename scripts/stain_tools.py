from __future__ import division

import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

def read_image(path):
    """
    Read an image to RGB uint8.
    Read with opencv (cv) and covert from BGR colorspace to RGB.

    :param path: The path to the image.
    :return: RGB uint8 image.
    """
    assert os.path.isfile(path), 'File not found'
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


def show_colors(C):
    """
    Visualize rows of C as colors (RGB)

    :param C: An array N x 3 where the rows are considered as RGB colors.
    :return:
    """
    assert isinstance(C, np.ndarray)
    assert C.ndim == 2
    assert C.shape[1] == 3
    n = C.shape[0]
    range255 = C.max() > 1.0
    for i in range(n):
        if range255:
            plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=C[i] / 255, linewidth=20)
        else:
            plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=C[i], linewidth=20)
        plt.axis('off')
        plt.axis([0, 1, -1, n])


def show(image, now=True, fig_size=(10, 10)):
    """
    Show an image (np.array).
    Caution! Rescales image to be in range [0,1].

    :param image:
    :param now: plt.show() now?
    :param fig_size: Figure size.
    :return:
    """
    image = check_image(image)
    is_gray = True if image.ndim == 2 else False
    image = image.astype(np.float32)
    m, M = image.min(), image.max()
    if fig_size != None:
        plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    if is_gray:
        plt.imshow((image - m) / (M - m), cmap='gray')
    else:
        plt.imshow((image - m) / (M - m))
    plt.axis('off')
    if now == True:
        plt.show()


def build_stack(images):
    """
    Build a stack of images from a tuple/list of images.

    :param images: A tuple/list of images.
    :return:
    """
    N = len(images)
    images = [check_image(image) for image in images]
    for image in images:
        assert image.ndim == images[0].ndim
    is_gray = True if images[0].ndim == 2 else False
    if is_gray:
        h, w = images[0].shape
        stack = np.zeros((N, h, w))
    else:
        h, w, c = images[0].shape
        stack = np.zeros((N, h, w, c))
    for i in range(N):
        stack[i] = images[i]
    return stack


def patch_grid(ims, width=5, sub_sample=False, rand=False, save_name=None):
    """
    Display a grid of patches.

    :param ims: A patch 'stack'
    :param width: Images per row.
    :param sub_sample: Should we take a subsample?
    :param rand: Randomize subsample?
    :return:
    """
    N0 = np.shape(ims)[0]
    if sub_sample and rand:
        N = sub_sample
        idx = np.random.choice(range(N), sub_sample, replace=False)
        stack = ims[idx]
    elif sub_sample and not rand:
        N = sub_sample
        stack = ims[:N]
    else:
        N = N0
        stack = ims
    height = np.ceil(float(N) / width).astype(np.uint16)
    plt.rcParams['figure.figsize'] = (18, (18 / width) * height)
    plt.figure()
    for i in range(N):
        plt.subplot(height, width, i + 1)
        show(stack[i], now=False, fig_size=None)
    if save_name != None:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        plt.savefig(save_name)
    plt.show()


def standardize_brightness(I, percentile=95):
    """
    Standardize brightness.

    :param I: Image uint8 RGB.
    :return: Image uint8 RGB with standardized brightness.
    """
    assert is_uint8_image(I)
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0]
    p = np.percentile(L, percentile)
    I_LAB[:, :, 0] = np.clip(255. * L / p, 0, 255).astype(np.uint8)  # 255. float seems to be important...
    I = cv.cvtColor(I_LAB, cv.COLOR_LAB2RGB)
    return I


def remove_zeros(I):
    """
    Remove zeros in an image, replace with 1's.

    :param I: An Array.
    :return: New array where 0's have been replaced with 1's.
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    I = remove_zeros(I)  # we don't want to take the log of zero..
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB
    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, 'Negative optical density'
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize the rows of an array.

    :param A: An array.
    :return: Array with rows normalized.
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'.
    Specifically, a pixel is not white if its luminance (in LAB color space) is less than the specified threshold.

    :param I: RGB uint 8 image.
    :param thresh: Luminosity threshold.
    :return: Binary mask where true denotes 'not white'.
    """
    assert is_uint8_image(I)
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)


def sign(x):
    """
    Returns the sign of x.

    :param x: A scalar x.
    :return: The sign of x  \in (+1, -1, 0).
    """
    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


### Checks

def array_equal(A, B, eps=1e-9):
    """
    Are arrays A and B equal?

    :param A: Array.
    :param B: Array.
    :param eps: Tolerance.
    :return: True/False.
    """
    if A.ndim != B.ndim:
        return False
    if A.shape != B.shape:
        return False
    if np.mean(A - B) > eps:
        return False
    return True


def is_image(x):
    """
    Is x an image?
    i.e. numpy array of 2 or 3 dimensions.

    :param x: Input.
    :return: True/False.
    """
    if not isinstance(x, np.ndarray):
        return False
    if x.ndim not in [2, 3]:
        return False
    return True


def is_gray_image(x):
    """
    Is x a gray image?

    :param x: Input.
    :return: True/False.
    """
    if not is_image(x):
        return False
    squeezed = x.squeeze()
    if not squeezed.ndim == 2:
        return False
    return True


def is_uint8_image(x):
    """
    Is x a uint8 image?

    :param x: Input.
    :return: True/False.
    """
    if not is_image(x):
        return False
    if x.dtype != np.uint8:
        return False
    return True


def check_image(x):
    """
    Check if is an image.
    If gray make sure it is 'squeezed' correctly.

    :param x: Input.
    :return: True/False.
    """
    assert is_image(x)
    if is_gray_image(x):
        x = x.squeeze()
    return x
# Defined in utils/misc_utils
def standardize_brightness(I, percentile=95):
    """
    Standardize brightness
    :param I:
    :return:
    """
    assert is_uint8_image(I)
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0]
    p = np.percentile(L, percentile)
    I_LAB[:, :, 0] = np.clip(255. * L / p, 0, 255).astype(np.uint8)  # 255. float seems to be important...
    I = cv.cvtColor(I_LAB, cv.COLOR_LAB2RGB)
    return I



class Normaliser(object):
    """
    Abstract base class for normalizers. Defines some necessary methods to be considered a normalizer.
    """

    def __init__(self, **kwargs):
        self.standardize = kwargs['standardize'] if 'standardize' in kwargs.keys() else True
        if self.standardize:
            print('Using brightness standardization')
        else:
            print('Not standardizing brightness')

    #@abstractmethod
    def fit(self, target):
        """Fit the normalizer to an target image"""

    #@abstractmethod
    def transform(self, I):
        """Transform an image to the target stain"""



class ReinhardNormalizer(Normaliser):


    def __init__(self, **kwargs):
        super(ReinhardNormalizer, self).__init__(**kwargs)
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        """
        Fit to a target image

        :param target: Image RGB uint8.
        :return:
        """
        if self.standardize:
            target = standardize_brightness(target)
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        """
        Transform an image.

        :param I: Image RGB uint8.
        :return:
        """
        if self.standardize:
            I = standardize_brightness(I)
        I1, I2, I3 = self.lab_split(I)
        means, stds = self.get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return self.merge_back(norm1, norm2, norm3)

    @staticmethod
    def lab_split(I):
        """
        Convert from RGB uint8 to LAB and split into channels.

        :param I: Image RGB uint8.
        :return:
        """
        assert is_uint8_image(I)
        I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        I = I.astype(np.float32)
        I1, I2, I3 = cv.split(I)
        I1 /= 2.55
        I2 -= 128.0
        I3 -= 128.0
        return I1, I2, I3

    @staticmethod
    def merge_back(I1, I2, I3):
        """
        Take seperate LAB channels and merge back to give RGB uint8.

        :param I1: L
        :param I2: A
        :param I3: B
        :return: Image RGB uint8.
        """
        I1 *= 2.55
        I2 += 128.0
        I3 += 128.0
        I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
        return cv.cvtColor(I, cv.COLOR_LAB2RGB)

    def get_mean_std(self, I):
        """
        Get mean and standard deviation of each channel.

        :param I: Image RGB uint8.
        :return:
        """
        I1, I2, I3 = self.lab_split(I)
        m1, sd1 = cv.meanStdDev(I1)
        m2, sd2 = cv.meanStdDev(I2)
        m3, sd3 = cv.meanStdDev(I3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds

