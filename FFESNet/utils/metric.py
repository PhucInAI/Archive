"""
    Metrics for evaluate images
"""


# pylint: disable=C0103, W0201, W0107, C0325

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
from scipy.ndimage import convolve, center_of_mass, distance_transform_edt as dtedt


def read_mask(path: str) -> tf.Tensor:
    """
    Reads and preprocesses a mask image from a file path.

    Args:
        path (str): The file path to the mask image.

    Returns:
        tf.Tensor: The preprocessed mask tensor.
    """
    mask_raw = tf.io.read_file(path)
    mask = tf.io.decode_jpeg(mask_raw, channels=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.image.resize(mask, [352, 352])
    mask = mask / 255.0
    mask = tf.expand_dims(mask, axis=0)

    return mask


def dice_coef(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the Dice coefficient between the ground truth mask and the predicted mask.

    Args:
        y_mask (tf.Tensor): Ground truth mask tensor.
        y_pred (tf.Tensor): Predicted mask tensor.

    Returns:
        tf.Tensor: The Dice coefficient value.
    """
    smooth = 1e-15

    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    y_mask = tf.cast(tf.math.greater(y_mask, 0.5), tf.float32)

    intersection = tf.reduce_sum(tf.multiply(y_mask, y_pred), axis=(1, 2, 3))
    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2, 3)) + smooth
    dice = tf.reduce_mean((2 * intersection + smooth) / union)

    return dice


def iou_metric(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the Intersection over Union (IoU) metric between the ground truth mask and the predicted mask.

    Args:
        y_mask (tf.Tensor): Ground truth mask tensor.
        y_pred (tf.Tensor): Predicted mask tensor.

    Returns:
        tf.Tensor: The IoU value.
    """
    smooth = 1e-15

    y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
    y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)

    intersection = tf.reduce_sum(tf.multiply(y_mask, y_pred), axis=(1, 2))
    union = tf.reduce_sum((y_mask + y_pred), axis=(1, 2)) + smooth

    iou = tf.reduce_mean(intersection / (union - intersection))

    return iou


def MAE(y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the Mean Absolute Error (MAE) between the ground truth mask and the predicted mask.

    Args:
        y_mask (tf.Tensor): Ground truth mask tensor.
        y_pred (tf.Tensor): Predicted mask tensor.

    Returns:
        tf.Tensor: The MAE value.
    """
    return tf.reduce_mean(tf.abs(y_pred - y_mask))


class WFbetaMetric(object):
    """
    Computes the Weighted F-measure (wFb) metric for evaluating foreground maps.

    Reference:
        How to Evaluate Foreground Maps? (CVPR 2014)
    """
    def __init__(self, beta: int = 1) -> None:
        """
        Initializes the WFbetaMetric class.

        Args:
            beta (int): The beta parameter for the metric. Default is 1.
        """
        self.beta = beta
        self.eps = 1e-12

    def _gaussian_distribution(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        Computes the Gaussian distribution for a given array.

        Args:
            x (np.ndarray): The input array.
            mu (float): Mean of the Gaussian distribution.
            sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
            np.ndarray: Gaussian distribution of the input array.
        """
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
            -np.power((x - mu) / sigma, 2) / 2)

    def _generate_gaussian_kernel(self, size: int, sigma: float = 1.0, mu: float = 0.0) -> np.ndarray:
        """
        Generates a 2D Gaussian kernel.

        Args:
            size (int): Size of the kernel (size x size).
            sigma (float): Standard deviation of the Gaussian distribution.
            mu (float): Mean of the Gaussian distribution.

        Returns:
            np.ndarray: The generated Gaussian kernel.
        """
        self.kernel_1d = np.linspace(-(size // 2), size // 2, size)
        self.kernel_1d = self._gaussian_distribution(self.kernel_1d, mu, sigma)
        self.kernel_2d = np.outer(self.kernel_1d.T, self.kernel_1d)
        self.kernel_2d *= 1.0 / self.kernel_2d.max()
        return self.kernel_2d

    def __call__(self, y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the WFbeta metric.

        Args:
            y_mask (tf.Tensor): Ground truth mask tensor.
            y_pred (tf.Tensor): Predicted mask tensor.

        Returns:
            tf.Tensor: The WFbeta metric value.
        """
        assert y_pred.ndim == y_mask.ndim and y_pred.shape == y_mask.shape
        y_mask = tf.squeeze(y_mask)
        y_pred = tf.squeeze(y_pred)

        y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.int32)
        y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.int32)
        y_pred = y_pred.numpy()
        y_mask = y_mask.numpy()

        Dst, Idxt = dtedt(y_mask == 0, return_indices=True)
        E = np.abs(y_pred - y_mask)
        Et = np.copy(E)
        Et[y_mask == 0] = Et[Idxt[0][y_mask == 0], Idxt[1][y_mask == 0]]

        K = self._generate_gaussian_kernel(size=7, sigma=5.0)
        EA = convolve(Et, weights=K, mode='constant', cval=0)
        MIN_E_EA = np.where(y_mask & (EA < E), EA, E)

        B = np.where(y_mask == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(y_mask))
        Ew = MIN_E_EA * B

        TPw = np.sum(y_mask) - np.sum(Ew[y_mask == 1])
        FPw = np.sum(Ew[y_mask == 0])

        R = 1 - np.mean(Ew[y_mask])
        P = TPw / (self.eps + TPw + FPw)

        wfb = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)

        return tf.cast(wfb, dtype=tf.float32)


class SMeasure(object):
    """
    Computes the S-measure for evaluating foreground maps.

    Reference:
        Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    """
    def __init__(self, alpha: float = 0.5) -> None:
        """
        Initializes the SMeasure class.

        Args:
            alpha (float): The alpha parameter for the metric. Default is 0.5.
        """
        self.alpha = alpha

    def _object(self, inp1: np.ndarray, inp2: np.ndarray) -> tf.Tensor:
        """
        Computes the object-level similarity score.

        Args:
            inp1 (np.ndarray): First input array (foreground).
            inp2 (np.ndarray): Second input array (ground truth).

        Returns:
            tf.Tensor: The object-level similarity score.
        """
        x = np.mean(inp1[inp2])
        sigma_x = np.std(inp1[inp2])
        score = 2 * x / (x**2 + 1 + sigma_x + 1e-8)
        return tf.cast(score, dtype=tf.float32)

    def s_object(self, SM: tf.Tensor, GT: tf.Tensor) -> tf.Tensor:
        """
        Computes the object-level similarity between the predicted and ground truth masks.

        Args:
            SM (tf.Tensor): Predicted mask tensor.
            GT (tf.Tensor): Ground truth mask tensor.

        Returns:
            tf.Tensor: The object-level similarity score.
        """
        fg = SM * GT
        bg = (1 - SM) * (1 - GT)

        u = tf.reduce_mean(GT)
        GT = tf.cast(GT, dtype=tf.bool)
        return u * self._object(fg.numpy(), GT.numpy()) + (1 - u) * self._object(bg.numpy(), tf.logical_not(GT.numpy()))

    def _ssim(self, SM: tf.Tensor, GT: tf.Tensor) -> tf.Tensor:
        """
        Computes the Structural Similarity Index (SSIM) between two tensors.

        Args:
            SM (tf.Tensor): First tensor (predicted mask).
            GT (tf.Tensor): Second tensor (ground truth mask).

        Returns:
            tf.Tensor: The SSIM score.
        """
        h, w = SM.shape
        N = h * w

        x = tf.reduce_mean(SM)
        y = tf.reduce_mean(GT)

        sigma_x = tf.math.reduce_variance(SM)
        sigma_y = tf.math.reduce_variance(GT)
        sigma_xy = tf.reduce_sum((SM - x) * (GT - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score

    def _divideGT(self, GT: tf.Tensor, x: int, y: int) -> tuple:
        """
        Divides the ground truth mask into four blocks.

        Args:
            GT (tf.Tensor): Ground truth mask tensor.
            x (int): x-coordinate for division.
            y (int): y-coordinate for division.

        Returns:
            tuple: Four blocks of the ground truth mask and their respective weights.
        """
        h, w = GT.shape
        area = h * w
        UL = GT[0:y, 0:x]
        UR = GT[0:y, x:w]
        LL = GT[y:h, 0:x]
        LR = GT[y:h, x:w]

        w1 = (x * y) / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return UL, UR, LL, LR, w1, w2, w3, w4

    def _divideSM(self, SM: tf.Tensor, x: int, y: int) -> tuple:
        """
        Divides the predicted mask into four blocks.

        Args:
            SM (tf.Tensor): Predicted mask tensor.
            x (int): x-coordinate for division.
            y (int): y-coordinate for division.

        Returns:
            tuple: Four blocks of the predicted mask.
        """
        h, w = SM.shape
        UL = SM[0:y, 0:x]
        UR = SM[0:y, x:w]
        LL = SM[y:h, 0:x]
        LR = SM[y:h, x:w]

        return UL, UR, LL, LR

    def s_region(self, y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the region-aware structural similarity score.

        Args:
            y_mask (tf.Tensor): Ground truth mask tensor.
            y_pred (tf.Tensor): Predicted mask tensor.

        Returns:
            tf.Tensor: The region-aware structural similarity score.
        """
        [y, x] = center_of_mass(y_mask.numpy())
        x = int(round(x)) + 1
        y = int(round(y)) + 1

        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(GT=y_mask, x=x, y=y)
        sm1, sm2, sm3, sm4 = self._divideSM(SM=y_pred, x=x, y=y)

        score1 = self._ssim(sm1, gt1)
        score2 = self._ssim(sm2, gt2)
        score3 = self._ssim(sm3, gt3)
        score4 = self._ssim(sm4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def __call__(self, y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the S-measure metric.

        Args:
            y_mask (tf.Tensor): Ground truth mask tensor.
            y_pred (tf.Tensor): Predicted mask tensor.

        Returns:
            tf.Tensor: The S-measure value.
        """
        assert y_pred.ndim == y_mask.ndim and y_pred.shape == y_mask.shape
        y_mask = tf.squeeze(y_mask)
        y_pred = tf.squeeze(y_pred)
        y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
        y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)
        y = tf.reduce_mean(y_mask)
        if y == 0:
            score = 1 - tf.reduce_mean(y_pred)
        elif y == 1:
            score = tf.reduce_mean(y_pred)
        else:
            score = self.alpha * self.s_object(y_pred, y_mask) + (1 - self.alpha) * self.s_region(y_mask=y_mask, y_pred=y_pred)
        return score


class Emeasure(object):
    """
    Computes the Enhanced-alignment measure for evaluating binary foreground maps.

    Reference:
        Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    """
    def __init__(self) -> None:
        """
        Initializes the Emeasure class.
        """
        pass

    def AlignmentTerm(self, dFM: np.ndarray, dy_mask: np.ndarray) -> np.ndarray:
        """
        Computes the alignment term between the predicted foreground map and the ground truth mask.

        Args:
            dFM (np.ndarray): Predicted foreground map.
            dy_mask (np.ndarray): Ground truth mask.

        Returns:
            np.ndarray: The alignment matrix.
        """
        mu_FM = np.mean(dFM)
        mu_y_mask = np.mean(dy_mask)
        align_FM = dFM - mu_FM
        align_y_mask = dy_mask - mu_y_mask
        align_Matrix = 2. * (align_y_mask * align_FM) / (
            align_y_mask * align_y_mask + align_FM * align_FM + 1e-8)
        return align_Matrix

    def EnhancedAlignmentTerm(self, align_Matrix: np.ndarray) -> np.ndarray:
        """
        Computes the enhanced alignment term.

        Args:
            align_Matrix (np.ndarray): The alignment matrix.

        Returns:
            np.ndarray: The enhanced alignment matrix.
        """
        enhanced = np.power(align_Matrix + 1, 2) / 4
        return enhanced

    def __call__(self, y_mask: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the E-measure metric.

        Args:
            y_mask (tf.Tensor): Ground truth mask tensor.
            y_pred (tf.Tensor): Predicted mask tensor.

        Returns:
            tf.Tensor: The E-measure value.
        """
        assert y_pred.ndim == y_mask.ndim and y_pred.shape == y_mask.shape
        y_mask = tf.squeeze(y_mask)
        y_pred = tf.squeeze(y_pred)
        y_pred = tf.cast(tf.greater(y_pred, 0.5), dtype=tf.float32)
        y_mask = tf.cast(tf.greater(y_mask, 0.5), dtype=tf.float32)

        y_mask = y_mask.numpy()
        y_pred = y_pred.numpy()

        th = 2 * y_pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(y_mask.shape)
        FM[y_pred >= th] = 1
        FM = np.array(FM, dtype=bool)
        y_mask = np.array(y_mask, dtype=bool)
        dFM = np.double(FM)

        if (sum(sum(np.double(y_mask))) == 0):
            enhanced_matrix = 1.0 - dFM
        elif (sum(sum(np.double(~y_mask))) == 0):
            enhanced_matrix = dFM
        else:
            dy_mask = np.double(y_mask)
            align_matrix = self.AlignmentTerm(dFM, dy_mask)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)

        [w, h] = np.shape(y_mask)
        score = sum(sum(enhanced_matrix)) / (w * h - 1 + 1e-8)

        return tf.cast(score, dtype=tf.float32)

def calculate_image_pair(gt_path, pred_path):
    """Calculate all metrics for 1 GT-pred pair"""

    wFb_obj = WFbetaMetric()
    smeasure_obj = SMeasure()
    emeasure_obj = Emeasure()

    y_mask = read_mask(gt_path)
    y_pred = read_mask(pred_path)

    dice_val = dice_coef(y_mask, y_pred).numpy()
    iou_val = iou_metric(y_mask, y_pred).numpy()
    wFb_val = wFb_obj(y_mask=y_mask, y_pred=y_pred).numpy()
    smeasure_val = smeasure_obj(y_mask=y_mask, y_pred=y_pred).numpy()
    emeasure_val = emeasure_obj(y_mask=y_mask, y_pred=y_pred).numpy()

    return dice_val, iou_val, wFb_val, smeasure_val, emeasure_val
