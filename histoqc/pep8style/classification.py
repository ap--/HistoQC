from histoqc.ClassificationModule import byExampleWithFeatures as by_example_with_features
from histoqc.ClassificationModule import compute_features as compute_features
from histoqc.ClassificationModule import compute_frangi as compute_frangi
from histoqc.ClassificationModule import compute_gabor as compute_gabor
from histoqc.ClassificationModule import compute_gaussian as compute_gaussian
from histoqc.ClassificationModule import compute_laplace as compute_laplace
from histoqc.ClassificationModule import compute_lbp as compute_lbp
from histoqc.ClassificationModule import compute_median as compute_median
from histoqc.ClassificationModule import compute_rgb as compute_rgb
from histoqc.ClassificationModule import pixelWise as pixel_wise

__all__ = [
    'pixel_wise',
    'compute_rgb',
    'compute_laplace',
    'compute_lbp',
    'compute_gaussian',
    'compute_median',
    'compute_gabor',
    'compute_frangi',
    'compute_features',
    'by_example_with_features'
]
