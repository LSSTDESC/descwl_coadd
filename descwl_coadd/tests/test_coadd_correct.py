import numpy as np
import galsim

import pytest

from descwl_shear_sims import SEObs

from ..coadd import MultiBandCoadds


def _plot_cmp(coadd_img, true_coadd_img, rtol, atol, crazy_obj, crazy_wcs, name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axs = plt.subplots(nrows=2, ncols=2)
    sns.heatmap(true_coadd_img, ax=axs[0, 0])
    axs[0, 0].set_title("true image")
    sns.heatmap(coadd_img, ax=axs[0, 1])
    axs[0, 1].set_title("coadded image")
    sns.heatmap(coadd_img - true_coadd_img, ax=axs[1, 0])
    axs[1, 0].set_title("coadded - true")
    sns.heatmap(
        (np.abs(coadd_img - true_coadd_img) <
         atol + rtol * np.abs(true_coadd_img)).astype(int),
        ax=axs[1, 1],
        vmin=0, vmax=1)
    axs[1, 1].set_title("pass vs fail at %s" % atol)
    plt.tight_layout()
    plt.savefig("coadd_%s_test_wcs%s_obj%s.png" % (name, crazy_wcs, crazy_obj))
    plt.close()


@pytest.mark.parametrize('crazy_obj', [False, True])
@pytest.mark.parametrize('crazy_wcs', [False, True])
def test_coadd_image_correct(crazy_wcs, crazy_obj):
    n_coadd = 10
    coadd_dim = 53
    coadd_cen = (coadd_dim + 1) / 2
    se_dim = int(np.ceil(coadd_dim * np.sqrt(2)))
    if se_dim % 2 == 0:
        se_dim += 1
    se_cen = (se_dim + 1) / 2
    scale = 0.2
    noise_std = 0.1
    world_origin = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)

    aff = galsim.PixelScale(scale).affine()
    aff = aff.withOrigin(galsim.PositionD(coadd_cen, coadd_cen), galsim.PositionD(0, 0))
    coadd_wcs = galsim.TanWCS(
        aff,
        world_origin,
    )

    rng = np.random.RandomState(seed=42)

    def _gen_psf_func(wcs, fwhm):
        def _psf_func(*args, **kargs):
            return galsim.Gaussian(fwhm=fwhm).drawImage(
                nx=101, ny=101, wcs=wcs.local(world_pos=world_origin)
            ), galsim.PositionD(0, 0)
        return _psf_func

    se_obs = []
    wgts = []
    objs = []
    psf_objs = []
    for _ in range(n_coadd):
        if crazy_obj:
            _fwhm = 2.9 * (1.0 + rng.normal() * 0.1)
            _g1 = rng.normal() * 0.3
            _g2 = rng.normal() * 0.3
            obj = galsim.Gaussian(fwhm=_fwhm).shear(g1=_g1, g2=_g2)
        else:
            obj = galsim.Gaussian(fwhm=2.9).shear(g1=-0.1, g2=0.3)

        objs.append(obj)

        if crazy_wcs:
            shear = galsim.Shear(g1=rng.normal() * 0.01, g2=rng.normal() * 0.01)
            aff = galsim.ShearWCS(scale, shear).affine()
            aff = aff.withOrigin(
                galsim.PositionD(se_cen, se_cen), galsim.PositionD(0, 0))
            wcs = galsim.TanWCS(
                aff,
                world_origin,
            )
        else:
            aff = galsim.PixelScale(scale).affine()
            aff = aff.withOrigin(
                galsim.PositionD(se_cen, se_cen), galsim.PositionD(0, 0))
            wcs = galsim.TanWCS(
                aff,
                world_origin,
            )

        _noise = noise_std * (1 + (rng.uniform() - 0.5)*2*0.05)
        wgts.append(1.0 / _noise**2)

        wgt = galsim.ImageD(np.ones((se_dim, se_dim)) / _noise**2)
        nse = galsim.ImageD(rng.normal(size=(se_dim, se_dim)) * _noise)
        bmsk = galsim.ImageI(np.zeros((se_dim, se_dim)))
        omsk = galsim.ImageI(np.zeros((se_dim, se_dim)))

        img = obj.drawImage(
            nx=se_dim,
            ny=se_dim,
            wcs=wcs.local(world_pos=world_origin),
        )

        if crazy_obj:
            _psf_fwhm = 1.0 * (1.0 + rng.normal() * 0.1)
        else:
            _psf_fwhm = 1.0
        psf_objs.append(galsim.Gaussian(fwhm=_psf_fwhm))

        se_obs.append(
            SEObs(
                image=img,
                weight=wgt,
                wcs=wcs,
                psf_function=_gen_psf_func(wcs, _psf_fwhm),
                noise=nse,
                bmask=bmsk,
                ormask=omsk,
            )
        )

    mbcoadd = MultiBandCoadds(
        data={"i": se_obs},
        coadd_wcs=coadd_wcs,
        coadd_dims=(coadd_dim, coadd_dim),
        psf_dims=(53, 53),
        byband=False,
        loglevel='debug',
        rng=np.random.RandomState(seed=10),
    )

    coadd = mbcoadd.get_coadd()
    coadd_img = coadd.image
    coadd_psf = coadd.psf.image

    wgts = np.array(wgts) / np.sum(wgts)
    true_coadd_img = galsim.Sum(
        [obj.withFlux(wgt) for obj, wgt in zip(objs, wgts)]
    ).drawImage(
        nx=coadd_dim,
        ny=coadd_dim,
        wcs=coadd_wcs.local(world_pos=world_origin)).array

    true_coadd_psf = galsim.Sum(
        [obj.withFlux(wgt) for obj, wgt in zip(psf_objs, wgts)]
    ).drawImage(
        nx=53,
        ny=53,
        wcs=coadd_wcs.local(world_pos=world_origin)).array

    if not crazy_wcs:
        rtol = 0
        atol = 5e-7
    else:
        rtol = 0
        atol = 5e-5

    coadd_img_err = np.max(np.abs(coadd_img - true_coadd_img))
    coadd_psf_err = np.max(np.abs(coadd_psf - true_coadd_psf))
    print("image max abs error:", coadd_img_err)
    print("psf max abs error:", coadd_psf_err)

    if not np.allclose(coadd_img, true_coadd_img, rtol=rtol, atol=atol):
        _plot_cmp(coadd_img, true_coadd_img, rtol, atol, crazy_obj, crazy_wcs, "img")

    if not np.allclose(coadd_psf, true_coadd_psf, rtol=rtol, atol=atol):
        _plot_cmp(coadd_psf, true_coadd_psf, rtol, atol, crazy_obj, crazy_wcs, "psf")

    assert np.allclose(coadd_img, true_coadd_img, rtol=rtol, atol=atol)
    assert np.allclose(coadd_psf, true_coadd_psf, rtol=rtol, atol=atol)
    assert np.all(np.isfinite(coadd.noise))
