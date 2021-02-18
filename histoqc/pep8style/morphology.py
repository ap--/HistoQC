from histoqc.MorphologyModule import fillSmallHoles as fill_small_holes
from histoqc.MorphologyModule import removeFatlikeTissue as remove_fatlike_tissue
from histoqc.MorphologyModule import removeSmallObjects as remove_small_objects
from histoqc.MorphologyModule import remove_large_objects as remove_large_objects

__all__ = [
    'remove_small_objects',
    'remove_large_objects',
    'remove_fatlike_tissue',
    'fill_small_holes'
]
