from albumentations.core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    ImageColorType,
    KeypointInternalType,
    ScaleFloatType,
    to_tuple,
)
import cv2
import numpy as np


# Slight modification of https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py#L1920
# to allow for drawing uniformly alpha and sigma params

from albumentations.augmentations.geometric import functional as F
import cv2
import random

from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration

class ElasticTransformDrawUniform(DualTransform):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Args:
        alpha (float):
        sigma (float): Gaussian filter parameter.
        alpha_affine (float): The range will be (-alpha_affine, alpha_affine)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        approximate (boolean): Whether to smooth displacement map with fixed kernel size.
                               Enabling this option gives ~2X speedup on large images.
        same_dxdy (boolean): Whether to use same random generated shift for x and y.
                             Enabling this option gives ~2X speedup.

    Targets:
        image, mask, bbox

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        alpha=(0.9, 1.1),
        sigma=(40, 60),
        alpha_affine=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        approximate=False,
        same_dxdy=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.approximate = approximate
        self.same_dxdy = same_dxdy
    
    
    def get_params(self):
        alpha = random.uniform(self.alpha[0], self.alpha[1])
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        
        return {
            "alpha": alpha,
            "sigma": sigma,
        }
        
    def apply(self, img, alpha, sigma, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
        return F.elastic_transform(
            img,
            alpha,
            sigma,
            self.alpha_affine,
            interpolation,
            self.border_mode,
            self.value,
            np.random.RandomState(random_state),
            self.approximate,
            self.same_dxdy,
        )

    def apply_to_mask(self, img, alpha, sigma, random_state=None, **params):
        return F.elastic_transform(
            img,
            alpha,
            sigma,
            self.alpha_affine,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_state),
            self.approximate,
            self.same_dxdy,
        )


    def get_transform_init_args_names(self):
        return (
            "alpha",
            "sigma",
            "alpha_affine",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
            "approximate",
            "same_dxdy",
        )
        

class StainAugmentation:

    def __init__(
        self,
        sigma2=(0.9, 1.1),
        out_indices=[0, 5, 11],
        always_apply=False,
        p=0.5,
    ):
        self.sigma2 = sigma2
        self.always_apply = always_apply
        self.out_indices = out_indices
        self.p = p
        
    
    def get_params(self,):
        return {
            "sigma2": random.uniform(*self.sigma2)
        }
    def __call__(self, image, mask):
        if (random.random() < self.p) or self.always_apply:
            params = self.get_params()
            mask_out = np.isin(mask, self.out_indices)

            # do this if there are pixels that are NOT background to normalize
            if not mask_out.all():            
                return rgb_perturb_stain_concentration(image, mask_out=mask_out, **params)
            
        return image
    
# class ReinhardNorm:
#     def __init__(
#         self,
#         mu: Tuple[float, float, float] = (8.74108109, -0.12440419,  0.0444982),
#         sigma: Tuple[float, float, float] = (0.6135447, 0.10989545, 0.0286032),
#         out_indices: List[int] = [0, 5, 11],
#     ):
#         self.mu = mu
#         self.sigma = simga
#         self.out_indices = out_indices
        
#         def __call__(self, image, mask):
#             tissue_rgb_normalized = reinhard(
#         tissue_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'],
#         mask_out=mask_out)