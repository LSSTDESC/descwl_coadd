import numpy as np

from descwl_coadd.interp import (
    replace_flag_with_noise, interp_image_nocheck,
)


def test_replace_flag_with_noise():
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
        mask=mask,
        flag=8,
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


def test_interpolate_image_and_noise_weight():
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 0
    weight[30:35, 40:45] = 0.0

    # put nans here to make sure interp is done ok
    msk = weight <= 0
    image[msk] = np.nan

    msk = (weight <= 0) | ((bmask & bad_flags) != 0)

    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    iimage = interp_image_nocheck(image=image, bad_msk=msk)
    inoise = interp_image_nocheck(image=noise, bad_msk=msk)

    assert np.allclose(iimage, 10 + x*5)

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[msk], inoise[msk])
    assert np.allclose(noise[~msk], inoise[~msk])


def test_interpolate_image_and_noise_bmask():
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 1

    rng = np.random.RandomState(seed=42)
    bmask[30:35, 40:45] = 1
    bmask[:, 0] = 2
    bmask[:, -1] = 4

    # put nans here to make sure interp is done ok
    msk = (weight <= 0) | ((bmask & bad_flags) != 0)
    image[msk] = np.nan

    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)

    iimage = interp_image_nocheck(image=image, bad_msk=msk)
    inoise = interp_image_nocheck(image=noise, bad_msk=msk)

    assert np.allclose(iimage, 10 + x*5)

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[msk], inoise[msk])
    assert np.allclose(noise[~msk], inoise[~msk])


def test_interpolate_image_and_noise_big_missing():
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 1

    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    bmask[15:80, 15:80] = 1

    # put nans here to make sure interp is done ok
    msk = (weight <= 0) | ((bmask & bad_flags) != 0)
    image[msk] = np.nan

    iimage = interp_image_nocheck(image=image, bad_msk=msk)
    inoise = interp_image_nocheck(image=noise, bad_msk=msk)

    # interp will be waaay off but shpuld have happened
    assert np.all(np.isfinite(iimage))

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[msk], inoise[msk])
    assert np.allclose(noise[~msk], inoise[~msk])


def test_interpolate_gauss_image(show=False):
    """
    test that our interpolation works decently for a linear
    piece missing from a gaussian image
    """

    noise = 0.001

    sigma = 4.0
    is2 = 1.0/sigma**2
    dims = 51, 51
    cen = (np.array(dims)-1.0)/2.0

    rows, cols = np.mgrid[
        0:dims[0],
        0:dims[1],
    ]
    rows = rows - cen[0]
    cols = cols - cen[1]

    image_unmasked = np.exp(-0.5*(rows**2 + cols**2)*is2)
    weight = image_unmasked*0 + 1.0/noise**2

    badcol = int(cen[1]-3)
    bw = 3
    rr = badcol-bw, badcol+bw+1

    weight[rr[0]:rr[1], badcol] = 0.0
    image_masked = image_unmasked.copy()
    image_masked[rr[0]:rr[1], badcol] = 0.0

    bmask = np.zeros_like(image_unmasked, dtype=np.int32)
    bad_flags = 0

    msk = (weight <= 0) | ((bmask & bad_flags) != 0)
    assert np.any(msk)

    iimage = interp_image_nocheck(image=image_masked, bad_msk=msk)

    maxdiff = np.abs(image_unmasked-iimage).max()

    if show:
        import images
        images.view_mosaic([image_masked, weight])

        images.compare_images(
            image_unmasked,
            iimage,
            width=2000,
            height=int(2000*2/3),
        )
        print('max diff:', maxdiff)

    assert maxdiff < 0.0015
