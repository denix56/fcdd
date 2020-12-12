import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import numpy as np
import random
from abc import abstractmethod
from typing import Callable
from kornia.filters import get_gaussian_kernel2d
import cv2
from PIL import Image
import skimage
from skimage import io, transform, morphology, segmentation, measure
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler


class TargetTransFunctor(object):
    def __init__(self, anomalous_label: int, outlier_classes: [int], nominal_label: int):
        self.anomalous_label = anomalous_label
        self.outlier_classes = outlier_classes
        self.nominal_label = nominal_label

    def __call__(self, x):
        return self.anomalous_label if x in self.outlier_classes else self.nominal_label


class AWGN(object):
    def __init__(self, std: float):
        self.std = std

    def __call__(self, x):
        return x + self.std * torch.randn_like(x)


def local_contrast_normalization_func(x):
    return local_contrast_normalization(x, scale='l1')


def local_contrast_normalization_func2(x):
    return local_contrast_normalization2(x, radius=3)


def remove_glare(x):
    img = np.asarray(x)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
    result = np.zeros_like(img)
    cv2.xphoto.inpaint(img, ~mask, result, cv2.xphoto.INPAINT_FSR_FAST)

    return Image.fromarray(result)


# def __find_main_red_line(img, mask):
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#
#     if not contours:
#         return mask
#     #
#     hierarchy = np.squeeze(hierarchy, 0)
#
#     contours_areas = [cv2.contourArea(cnt) for cnt in contours]
#     contours_rects = [cv2.boundingRect(cnt) for cnt in contours]
#     #
#     contours_areas_new = np.zeros((len(contours_areas)))
#     for i, area in enumerate(contours_areas):
#         child = hierarchy[i, 2]
#         child_area = 0
#         while child != -1:
#             child_area += contours_areas[child]
#             child = hierarchy[child, 0]
#         contours_areas_new[i] = area - child_area
#     contours_areas = contours_areas_new
#     #
#     contours_rects = [cnt for i, cnt in enumerate(contours_rects) if contours_areas[i] > 2000 and \
#                 contours_rects[i][2] == img.shape[1]]
#     contours_sorted = np.argsort(contours_areas)[::-1]
#     contours_sorted = contours_sorted[hierarchy[contours_sorted, 3] == -1]
#     #
#     if len(contours_sorted) > 1:
#         main_cnt = contours_sorted[0]
#
#
#         if count1 > 0 and contours_areas[contours_sorted[1]] > 2500:
#             child = hierarchy[contours_sorted[1], 2]
#             count2 = 0
#             while child != -1:
#                 count2 += 1
#                 child = hierarchy[child, 0]
#
#             if count2 < count1:
#                 main_cnt = contours_sorted[1]
#
#     else:
#         main_cnt = contours_sorted[0]
#
#     # mask = np.zeros_like(mask)
#     # for (x, y, w, h) in contours_rects:
#     #     mask[y:y+h, ...] = 255
#
#     # contours = [cnt for i, cnt in enumerate(contours) if cv2.boundingRect(cnt)]
#     # cv2.drawContours(mask, contours, -1, (255), -1)

# https://www.kaggle.com/icanfly/roi-cropping-and-specular-reflections-removing?scriptVersionId=1243073
def find_roi_by_gsmix(img, mask=None):
    h, w, _ = img.shape
    x_coor = np.repeat(range(h), w)  # for calculating the R
    y_coor = np.tile(range(w), h)
    if mask is None:
        center = [h / 2, w / 2]
    else:
        mask = mask.reshape(-1)
        center = [np.mean(x_coor[mask == 1]), np.mean(y_coor[mask == 1])]
    R = np.sqrt((x_coor - center[0]) ** 2 + (y_coor - center[1]) ** 2)  # R
    A = img[:, :, 1].reshape(-1)  # A
    #R = np.ones_like(A)
    Ra = np.vstack([R, A]).T  # concat R and A

    scaler = MinMaxScaler()
    Ra = scaler.fit_transform(Ra)
    gs_mix = GaussianMixture(n_components=2, random_state=42, init_params='kmeans')  # Gaussian mixture modele
    gs_mix.fit(Ra)
    labels = gs_mix.predict(Ra)

    # Cluster with lowest R-mean will be chosen as ROI
    means = gs_mix.means_
    if means[0, 0] < means[1, 0]:
        labels = 1 - labels
    mask = labels.reshape(h, w).astype(np.uint8)

    return mask


def postprocess_mask(mask, morp=False):
    selem = skimage.morphology.rectangle(2, 1)
    mask = skimage.morphology.binary_erosion(mask, selem)

    labels_mask = measure.label(mask)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    # n = 5
    # if len(regions) > n:
    #     for rg in regions[n:]:
    #         labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    # labels_mask[labels_mask != 0] = 1
    # mask = labels_mask

    # selem = skimage.morphology.rectangle(7, 3)
    # mask = skimage.morphology.binary_dilation(mask, selem)
    mask = skimage.morphology.remove_small_objects(mask > 0, 250, 2)

    if morp:
        selem = skimage.morphology.rectangle(20, 10)
        mask = skimage.morphology.binary_erosion(mask, selem)
        mask = skimage.morphology.binary_dilation(mask, selem)

    return mask

def remove_red_lines(x):
    img = np.asarray(x).copy()

    ## (1) Read and convert to HSV
    # lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # roi_mask = find_roi_by_gsmix(lab)
    # roi_mask = postprocess_mask(roi_mask, False).astype(np.uint8)
    # Image.fromarray(roi_mask*255).show()
    # img[roi_mask == 1] = 0
    #Image.fromarray(img).show()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))

    mask = mask1 | mask2
    mmm = mask.mean(axis=1) / 255
    mm = np.zeros(mmm.shape, dtype=np.bool)
    mmm = np.nonzero(mmm)
    mm[mmm[0][0]:mmm[0][-1]+1] = 1

    mm[:100] = 0
    mm[-100:] = 0
    mask = np.zeros_like(mask)
    mask[mm, :] = 1

    selem = skimage.morphology.rectangle(7, 3)
    mask = skimage.morphology.binary_dilation(mask, selem)

    img[mask > 0] = 0

    return Image.fromarray(img)


class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.clahe = None

    def __call__(self, x):
        img = np.asarray(x)

        is_rgb = len(img.shape) == 3 and img.shape[2] == 3

        if is_rgb:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab_planes = cv2.split(lab)

        if self.clahe is None:
            self.clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)

        if is_rgb:
            lab_planes[0] = self.clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            res = self.clahe.apply(img)

        return Image.fromarray(res)


def get_target_label_idx(labels: np.ndarray, targets: np.ndarray):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def local_contrast_normalization2(image: torch.tensor, radius=9):
    """
    image: torch.Tensor , .shape => (1,channels,height,width)

    radius: Gaussian filter size (int), odd
    """
    if radius % 2 == 0:
        radius += 1

    image = image.unsqueeze(0)

    n, c, h, w = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
    sigma = 2.0

    gaussian_filter = get_gaussian_kernel2d((radius, radius),
                                            (sigma, sigma)).unsqueeze(0).unsqueeze(1).to(image).expand(-1, c, -1, -1)
    filtered_out =  F.conv2d(image, gaussian_filter, padding=radius - 1)
    mid = int(np.floor(gaussian_filter.shape[2] / 2.))
    ### Subtractive Normalization
    centered_image = image - filtered_out[:, :, mid:-mid, mid:-mid]

    ## Variance Calc
    sum_sqr_image = F.conv2d(centered_image.pow(2), gaussian_filter, padding=radius - 1)
    s_deviation = sum_sqr_image[:, :, mid:-mid, mid:-mid].sqrt()
    per_img_mean = s_deviation.mean().item()

    ## Divisive Normalization
    per_img_mean = np.maximum(per_img_mean, 1e-4)
    divisor = s_deviation.clamp(min=per_img_mean)
    new_image = centered_image / divisor
    return new_image.squeeze(0)


def local_contrast_normalization(x: torch.tensor, scale: str = 'l2'):
    """
    Apply local contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale if x_scale != 0 else 1

    return x


class MultiCompose(transforms.Compose):
    """
    Like transforms.Compose, but applies all transformations to a multitude of variables, instead of just one.
    More importantly, for random transformations (like RandomCrop), applies the same choice of transformation, i.e.
    e.g. the same crop for all variables.
    """
    def __call__(self, imgs: []):
        for t in self.transforms:
            imgs = list(imgs)
            imgs = self.__multi_apply(imgs, t)
        return imgs

    def __multi_apply(self, imgs: [], t: Callable):
        if isinstance(t, transforms.RandomCrop):
            for idx, img in enumerate(imgs):
                if t.padding is not None and t.padding > 0:
                    img = TF.pad(img, t.padding, t.fill, t.padding_mode) if img is not None else img
                if t.pad_if_needed and img.size[0] < t.size[1]:
                    img = TF.pad(img, (t.size[1] - img.size[0], 0), t.fill, t.padding_mode) if img is not None else img
                if t.pad_if_needed and img.size[1] < t.size[0]:
                    img = TF.pad(img, (0, t.size[0] - img.size[1]), t.fill, t.padding_mode) if img is not None else img
                imgs[idx] = img
            i, j, h, w = t.get_params(imgs[0], output_size=t.size)
            for idx, img in enumerate(imgs):
                imgs[idx] = TF.crop(img, i, j, h, w) if img is not None else img
        elif isinstance(t, transforms.RandomHorizontalFlip):
            if random.random() > 0.5:
                for idx, img in enumerate(imgs):
                    imgs[idx] = TF.hflip(img)
        elif isinstance(t, transforms.RandomVerticalFlip):
            if random.random() > 0.5:
                for idx, img in enumerate(imgs):
                    imgs[idx] = TF.vflip(img)
        elif isinstance(t, transforms.ToTensor):
            for idx, img in enumerate(imgs):
                imgs[idx] = TF.to_tensor(img) if img is not None else None
        elif isinstance(
                t, (transforms.Resize, transforms.Lambda, transforms.ToPILImage, transforms.ToTensor, BlackCenter)
        ):
            for idx, img in enumerate(imgs):
                imgs[idx] = t(img) if img is not None else None
        elif isinstance(t, LabelConditioner):
            assert t.n == len(imgs)
            t_picked = t(*imgs)
            imgs[:-1] = self.__multi_apply(imgs[:-1], t_picked)
        elif isinstance(t, MultiTransform):
            assert t.n == len(imgs)
            imgs = t(*imgs)
        elif isinstance(t, transforms.RandomChoice):
            t_picked = random.choice(t.transforms)
            imgs = self.__multi_apply(imgs, t_picked)
        elif isinstance(t, MultiCompose):
            imgs = t(imgs)
        else:
            raise NotImplementedError('There is no multi compose version of {} yet.'.format(t.__class__))
        return imgs


class MultiTransform(object):
    """ Class to mark a transform operation that expects multiple inputs """
    n = 0  # amount of expected inputs
    pass


class ImgGTTargetTransform(MultiTransform):
    """ Class to mark a transform operation that expects three inputs: (image, ground-truth map, label) """
    n = 3
    @abstractmethod
    def __call__(self, img, gt, target):
        return img, gt, target


class ImgGtTransform(MultiTransform):
    """ Class to mark a transform operation that expects two inputs: (image, ground-truth map) """
    n = 2
    @abstractmethod
    def __call__(self, img, gt):
        return img, gt


class LabelConditioner(ImgGTTargetTransform):
    def __init__(self, conds: [int], t1: Callable, t2: Callable):
        """
        Applies transformation t1 if the encountered label is in conds, otherwise applies transformation t2.
        :param conds: list of labels
        :param t1: some transformation
        :param t2: some other transformation
        """
        self.conds = conds
        self.t1 = t1
        self.t2 = t2

    def __call__(self, img, gt, target):
        if target in self.conds:
            return self.t1
        else:
            return self.t2


class ImgTransformWrap(ImgGtTransform):
    """ Wrapper for some transformation that is used in a MultiCompose, but is only to be applied to the first input """
    def __init__(self, t: Callable):
        self.t = t

    def __call__(self, img, gt):
        return self.t(img), gt


class BlackCenter(object):
    def __init__(self, percentage: float = 0.5, inverse: bool = False):
        """
        Blackens the center of given image, i.e. puts pixel value to zero.
        :param percentage: the percentage of the center in the overall image.
        :param inverse: whether to inverse the operation, i.e. blacken the borders instead.
        """
        self.percentage = percentage
        self.inverse = inverse

    def __call__(self, img: torch.Tensor):
        c, h, w = img.shape
        oh, ow = int((1 - self.percentage) * h * 0.5), int((1 - self.percentage) * w * 0.5)
        if not self.inverse:
            img[:, oh:-oh, ow:-ow] = 0
        else:
            img[:, :oh, :] = 0
            img[:, -oh:, :] = 0
            img[:, :, :ow] = 0
            img[:, :, -ow:] = 0
        return img


