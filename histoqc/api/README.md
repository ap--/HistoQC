# histoqc.api

Provides a pep8 compatible and pep484 type annotated interface to the histoqc 
submodules. Configuration parameters are typed keyword arguments for the 
different histoqc functions.

## examples

```python

import histoqc.api as qc
from histoqc.api import PipelineChain, PipelineState

def make_config_first_chain() -> 'Callable[[PipelineState], np.ndarray]':
    """example how to build the call chain for config_first.ini"""
    chain = PipelineChain()
    qc.get_basic_stats(chain)
    qc.get_intensity_threshold_percent(
        chain,
        name="nonwhite",
        upper_threshold=0.9,
        lower_variance=10,
    )
    qc.compare_to_templates(
        chain,
        limit_to_mask=True,
        bins=20,
        templates=[
            "./templates/template1.png",
            "./templates/template2.png",
            "./templates/template3.png",
            "./templates/template4.png",
        ],
    )
    qc.get_histogram(
        chain,
        limit_to_mask=True,
        bins=20,
    )
    qc.get_contrast(
        chain,
        limit_to_mask=True,
    )
    qc.get_brightness_gray(
        chain,
        limit_to_mask=True,
    )
    qc.get_brightness_by_channel_in_color_space(
        chain,
        limit_to_mask=True,
        to_color_space="RGB",
    )
    qc.get_brightness_by_channel_in_color_space(
        chain,
        limit_to_mask=True,
        to_color_space="YUV",
    )
    qc.save_final_mask(chain)
    qc.save_thumbnails(
        chain,
        image_work_size="1.25x",
        small_dim=500,
    )
    qc.final_computations(chain)

    # return the constructed chain as callable
    return chain.run

my_qc_pipeline = make_config_first_chain()

ps = PipelineState.from_image(
    image_fn="/path/to/my/image.svs",
)

mask = my_qc_pipeline(ps)

```

## thoughts

Could switch to a different chain building pattern and make all functions 
methods on the chain...
