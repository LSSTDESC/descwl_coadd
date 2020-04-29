import numpy as np

from ..interp import replace_flag_with_noise


def test_replace_bright_with_noise():
    shape = (10, 10)
    err = 0.1
    rng = np.random.RandomState(seed=42)
    image = rng.normal(size=shape) * err * 0.0 + 2.0
    noise = rng.normal(size=shape) * err * 0.0 + 3.0
    weight = np.ones_like(image) / err / err
    mask = np.ones_like(image).astype(np.int32)
    xinds = [0, 4, 7, 8]
    yinds = [8, 3, 2, 6]
    mask[yinds, xinds] = 8
    mask[0, 2] = 8 * 2

    replace_flag_with_noise(
        rng=np.random.RandomState(seed=10),
        image=image,
        noise_image=noise,
        weight=weight,
        mask=mask
    )

    rng = np.random.RandomState(seed=42)
    oimage = rng.normal(size=shape) * err * 0.0 + 2.0
    onoise = rng.normal(size=shape) * err * 0.0 + 3.0

    assert not np.array_equal(oimage, image)
    assert not np.array_equal(onoise, noise)

    rng = np.random.RandomState(seed=10)
    w = np.where(oimage != image)
    assert w[0].size == 4
    assert set(w[0]) == set(yinds)
    assert set(w[1]) == set(xinds)
    w = np.where(oimage.ravel() != image.ravel())
    vals = rng.normal(size=w[0].size, scale=err)
    assert np.array_equal(image.ravel()[w], vals)

    w = np.where(onoise != noise)
    assert w[0].size == 4
    assert set(w[0]) == set(yinds)
    assert set(w[1]) == set(xinds)
    w = np.where(onoise.ravel() != noise.ravel())
    vals = rng.normal(size=w[0].size, scale=err)
    assert np.array_equal(noise.ravel()[w], vals)
