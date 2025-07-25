import sys
import os
import pytest
import numpy as np
import esutil as eu

from descwl_shear_sims.sim import make_sim, get_se_dim
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.stars import StarCatalog

from descwl_coadd.coadd import get_bad_mask, get_median_var, warp_exposures, warp_psf
from descwl_coadd.coadd import make_coadd_obs, make_coadd, make_coadd_old
from descwl_coadd.procflags import WARP_BOUNDARY
from descwl_coadd.interp import CTInterpolator
from descwl_shear_sims.galaxies import make_galaxy_catalog

from lsst.meas.algorithms.cloughTocher2DInterpolator import (
    CloughTocher2DInterpolateTask,
)

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def _make_sim(
    rng, psf_type,
    epochs_per_band=3,
    stars=False,
    dither=True,
    rotate=True,
    bands=['i', 'z'],
    coadd_dim=101,
    se_dim=None,
    psf_dim=51,
    bad_columns=False,
):
    buff = 5

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type="exp",
        coadd_dim=coadd_dim,
        buff=buff,
        layout="grid",
        gal_config={'mag': 22},
    )

    if se_dim is None:
        se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate)

    if psf_type == "ps":
        psf = make_ps_psf(rng=rng, dim=se_dim)
    else:
        psf = make_fixed_psf(psf_type=psf_type)

    if stars:
        star_catalog = StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=100,
        )
    else:
        star_catalog = None

    return make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        se_dim=se_dim,
        psf_dim=psf_dim,
        epochs_per_band=epochs_per_band,
        g1=0.02,
        g2=0.00,
        bands=bands,
        psf=psf,
        dither=dither,
        rotate=rotate,
        bad_columns=bad_columns,
    )


@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_coadds_smoke(dither, rotate):
    rng = np.random.RandomState(55)

    coadd_dim = 101
    psf_dim = 51

    bands = ['r', 'i', 'z']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
        dither=dither, rotate=rotate,
    )

    # coadd each band separately
    bdata = sim_data['band_data']
    for band in bands:
        assert band in bdata
        exps = bdata[band]

        coadd_data = make_coadd(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )

        for tp in ['exp', 'noise_exp', 'mfrac_exp']:
            name = f'coadd_{tp}'
            psf = coadd_data[name].getPsf()
            assert psf is not None

        coadd_dims = (coadd_dim, )*2
        psf_dims = (psf_dim, )*2
        assert coadd_data['coadd_exp'].image.array.shape == coadd_dims
        assert coadd_data['coadd_psf_exp'].image.array.shape == psf_dims

        assert np.all(np.isfinite(coadd_data['coadd_psf_exp'].image.array))
        assert np.all(coadd_data['coadd_mfrac_exp'].image.array == 0)


@pytest.mark.parametrize('buff', [4, 5])
@pytest.mark.parametrize('bad_columns', [False, True])
def test_coadds_with_interpolator_smoke(buff, bad_columns):
    rng = np.random.RandomState(123)

    coadd_dim = 101
    psf_dim = 51

    bands = ['r', 'i', 'z']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
        bad_columns=bad_columns,
    )

    interpolator = CTInterpolator(buff=buff)

    # coadd each band separately
    bdata = sim_data['band_data']
    for band in bands:
        assert band in bdata
        exps = bdata[band]

        coadd_data = make_coadd(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
            interpolator=interpolator,
        )

        for tp in ['exp', 'noise_exp', 'mfrac_exp']:
            name = f'coadd_{tp}'
            psf = coadd_data[name].getPsf()
            assert psf is not None

        coadd_dims = (coadd_dim, )*2
        psf_dims = (psf_dim, )*2
        assert coadd_data['coadd_exp'].image.array.shape == coadd_dims
        assert coadd_data['coadd_psf_exp'].image.array.shape == psf_dims

        assert np.all(np.isfinite(coadd_data['coadd_psf_exp'].image.array))

        if bad_columns:
            assert np.any(coadd_data['coadd_mfrac_exp'].image.array)
        else:
            assert np.all(coadd_data['coadd_mfrac_exp'].image.array == 0)


@pytest.mark.parametrize('flip_xy', [False, True])
@pytest.mark.parametrize('buff', [4, 5])
def test_coadd_equality_with_dm_interpolator(subtests, buff, flip_xy):
    """Test that the DM-implementation of the interpolator behaves identical.
    """
    rng = np.random.RandomState(123)

    coadd_dim = 101
    psf_dim = 51

    bands = ['r', 'i', 'z']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
        stars=False, bad_columns=True,
    )

    our_interpolator = CTInterpolator(buff=buff)

    config = CloughTocher2DInterpolateTask.ConfigClass()
    config.interpLength = buff
    config.badMaskPlanes = our_interpolator.bad_mask_planes
    config.flipXY = flip_xy

    dm_interpolator = CloughTocher2DInterpolateTask(config=config)

    # coadd each band separately
    bdata = sim_data['band_data']
    for band in bands:
        assert band in bdata
        exps = bdata[band]

        # Check that there are pixels to be interpolated.
        bit_mask = exps[0].mask.getPlaneBitMask(our_interpolator.bad_mask_planes)
        assert any((exp.mask.array & bit_mask).any() for exp in exps)

        # Since interpolation changes the exposures in place, make a deep
        # copy before making the coadds.
        coadd_data = make_coadd(
            exps=[exp.clone() for exp in exps],
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
            interpolator=our_interpolator,
        )

        dm_coadd_data = make_coadd(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
            interpolator=dm_interpolator,
        )

        # Compare only the main exposure since the noise realizations are
        # different and mfrac is not affected by the interpolator.
        if flip_xy:
            # Exact agreement is expected only with flipXY config set to True.
            np.testing.assert_array_equal(
                dm_coadd_data['coadd_exp'].image.array,
                coadd_data['coadd_exp'].image.array,
            )
        else:
            with np.testing.assert_raises(AssertionError):
                np.testing.assert_array_equal(
                    dm_coadd_data['coadd_exp'].image.array,
                    coadd_data['coadd_exp'].image.array,
                )
            # The agreement is not exact due to the different triangulation.
            # The test is intended to benchmark the disagreement due to a
            # different, but equally valid, choice of coordinates.
            # | actual - desired | < atol + rtol * |desired|
            with subtests.test(msg='Benchmark differences from DM interpolator'):
                np.testing.assert_allclose(
                    actual=dm_coadd_data['coadd_exp'].image.array,
                    desired=coadd_data['coadd_exp'].image.array,
                    atol=0.41,
                    rtol=0.0,
                )

                # Limit to pixels that weren't bad to begin with, but suffer from
                # warping the interpolated pixels nearby.
                msk = coadd_data['coadd_exp'].mask.array == 0
                np.testing.assert_allclose(
                    actual=dm_coadd_data['coadd_exp'].image.array[msk],
                    desired=coadd_data['coadd_exp'].image.array[msk],
                    atol=0.063,
                    rtol=0.0,
                )


@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_coadds_mfrac(dither, rotate):
    rng = np.random.RandomState(55)
    ntrial = 100

    coadd_dim = 101
    psf_dim = 51

    bands = ['r', 'i', 'z']

    for itrial in range(ntrial):
        sim_data = _make_sim(
            rng=rng, psf_type='gauss', bands=bands,
            coadd_dim=coadd_dim, psf_dim=psf_dim,
            dither=dither, rotate=rotate,
            bad_columns=True,
        )

        # coadd each band separately
        bdata = sim_data['band_data']
        for band in bands:
            assert band in bdata
            exps = bdata[band]

            coadd, exp_info = make_coadd_obs(
                exps=exps,
                coadd_wcs=sim_data['coadd_wcs'],
                coadd_bbox=sim_data['coadd_bbox'],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                remove_poisson=False,  # no object poisson noise in sims
            )

            coadd_dims = (coadd_dim, )*2
            psf_dims = (psf_dim, )*2
            assert coadd.image.shape == coadd_dims
            assert coadd.psf.image.shape == psf_dims

            assert np.all(np.isfinite(coadd.psf.image))
            assert np.all(coadd.mfrac >= 0)
            assert np.all(coadd.mfrac <= 1)

            # This depends on the realization, try until we get one
            if (
                np.any(coadd.mfrac != 0) and
                np.max(coadd.mfrac) > 0.1 and
                np.mean(coadd.mfrac) < 0.05
            ):
                ok = True
                break

            if False:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(coadd.mfrac)
                import pdb
                pdb.set_trace()
        if ok:
            break

    assert ok, 'mfrac not set properly'


@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_coadds_noise(dither, rotate):
    rng = np.random.RandomState(55)

    coadd_dim = 101
    psf_dim = 51

    cen = (coadd_dim - 1)//2
    pcen = (psf_dim - 1)//2

    bands = ['r', 'i', 'z']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
        dither=dither, rotate=rotate,
    )

    # coadd each band separately
    bdata = sim_data['band_data']
    for band in bands:
        assert band in bdata
        exps = bdata[band]

        coadd, exp_info = make_coadd_obs(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )

        coadd_dims = (coadd_dim, )*2
        psf_dims = (psf_dim, )*2
        assert coadd.image.shape == coadd_dims
        assert coadd.noise.shape == coadd_dims
        assert coadd.psf.image.shape == psf_dims

        assert np.all(np.isfinite(coadd.psf.image))
        assert np.all(coadd.mfrac == 0)

        emed = coadd.coadd_exp.variance.array[cen, cen]
        pmed = coadd.coadd_psf_exp.variance.array[pcen, pcen]
        nmed = coadd.coadd_noise_exp.variance.array[cen, cen]

        assert abs(pmed/emed-1) < 1.0e-3
        assert abs(nmed/emed-1) < 1.0e-3


@pytest.mark.parametrize('rotate', [False, True])
def test_coadds_boundary(rotate):
    ntrial = 3
    rng = np.random.RandomState(55)

    coadd_dim = 101
    se_dim = 111
    psf_dim = 51

    bands = ['i']
    epochs_per_band = 3

    if not rotate:
        ok = True
    else:
        ok = False

    for i in range(ntrial):
        sim_data = _make_sim(
            rng=rng, psf_type='gauss', bands=bands,
            coadd_dim=coadd_dim,
            se_dim=se_dim,
            psf_dim=psf_dim,
            rotate=rotate,
            epochs_per_band=epochs_per_band,
        )

        # coadd each band separately
        exps = sim_data['band_data'][bands[0]]

        coadd_dict = make_coadd(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )

        if not rotate:
            assert coadd_dict['nkept'] == epochs_per_band
        else:
            if coadd_dict['nkept'] != epochs_per_band:
                exp_info = coadd_dict['exp_info']
                wbad, = np.where(exp_info['flags'] != 0)
                ngood = exp_info.size - wbad.size
                assert ngood == coadd_dict['nkept']
                assert np.all(exp_info['flags'][wbad] & WARP_BOUNDARY != 0)

                ok = True
            break

    assert ok


def test_coadds_not_used():
    rng = np.random.RandomState(9312)

    epochs_per_band = 3
    sim_data = _make_sim(
        rng=rng,
        psf_type='gauss',
        bands=['i'],
        coadd_dim=101,
        psf_dim=51,
        epochs_per_band=epochs_per_band,
    )

    # TODO we don't have getId() yet, so assuming the first one is index 0 etc.
    # for now
    exps = sim_data['band_data']['i']
    exp = exps[1]
    flagval = exp.mask.getPlaneBitMask('BAD')

    exp.mask.array[:, :] = flagval

    coadd_dict = make_coadd(
        exps=exps,
        coadd_wcs=sim_data['coadd_wcs'],
        coadd_bbox=sim_data['coadd_bbox'],
        psf_dims=sim_data['psf_dims'],
        rng=rng,
        remove_poisson=False,  # no object poisson noise in sims
    )

    assert coadd_dict['nkept'] == epochs_per_band - 1
    exp_info = coadd_dict['exp_info']
    wbad, = np.where(exp_info['flags'] != 0)
    assert wbad.size == 1
    assert wbad[0] == 1


def test_new_coadds():
    seed = 9312

    for coadd_type in ('old', 'new'):
        rng = np.random.RandomState(seed)

        epochs_per_band = 3
        sim_data = _make_sim(
            rng=rng,
            psf_type='gauss',
            bands=['i'],
            coadd_dim=101,
            psf_dim=51,
            epochs_per_band=epochs_per_band,
        )

        # TODO we don't have getId() yet, so assuming the first one is index 0 etc.
        # for now
        exps = sim_data['band_data']['i']
        exp = exps[1]
        flagval = exp.mask.getPlaneBitMask('BAD')

        exp.mask.array[:, :] = flagval

        if coadd_type == 'new':
            function = make_coadd
        else:
            function = make_coadd_old

        coadd_dict = function(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )
        if coadd_type == 'new':
            cdnew = coadd_dict
        else:
            cdold = coadd_dict

    assert cdnew['nkept'] == cdold['nkept']
    assert eu.numpy_util.compare_arrays(cdnew['exp_info'], cdold['exp_info'])
    assert np.all(
        cdnew['coadd_exp'].image.array == cdold['coadd_exp'].image.array
    )
    assert np.all(
        cdnew['coadd_noise_exp'].image.array == cdold['coadd_noise_exp'].image.array
    )
    assert np.all(
        cdnew['coadd_psf_exp'].image.array == cdold['coadd_psf_exp'].image.array
    )
    assert np.all(
        cdnew['coadd_mfrac_exp'].image.array == cdold['coadd_mfrac_exp'].image.array
    )


@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_warped_exposures(dither, rotate):
    """Test that the warped exposure is same as a coadd with one input."""

    epochs_per_band = 1
    noise_seed = 9813

    rng = np.random.RandomState(1234)
    sim_data = _make_sim(
        rng=rng,
        psf_type='gauss',
        bands=['i'],
        coadd_dim=101,
        psf_dim=51,
        dither=dither,
        rotate=rotate,
        epochs_per_band=epochs_per_band,
    )

    exp = sim_data['band_data']['i'][0]
    medvar = get_median_var(exp, False)
    rng = np.random.RandomState(noise_seed)
    warp, noise_warp, mfrac_warp, _ = warp_exposures(
        exp=exp,
        coadd_wcs=sim_data['coadd_wcs'],
        coadd_bbox=sim_data['coadd_bbox'],
        rng=rng,
        remove_poisson=False,  # no object poisson noise in sims
    )

    _, maskfrac1 = get_bad_mask(exp)
    _, maskfrac2 = get_bad_mask(warp)
    # This is a trivial test of the maskfrac calculation.
    assert maskfrac1 == maskfrac2

    psf_warp = warp_psf(psf=exp.getPsf(), wcs=exp.getWcs(),
                        coadd_wcs=sim_data['coadd_wcs'],
                        coadd_bbox=sim_data['coadd_bbox'],
                        psf_dims=sim_data['psf_dims'],
                        var=medvar, filter_label=exp.getFilter())

    for make_coadd_function in (make_coadd, make_coadd_old,):
        rng = np.random.RandomState(noise_seed)
        coadd_dict = make_coadd_function(
            exps=sim_data['band_data']['i'],
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )

        np.testing.assert_array_equal(
            coadd_dict['coadd_exp'].image.array,
            warp.image.array
        )
        np.testing.assert_array_equal(
            coadd_dict['coadd_noise_exp'].image.array,
            noise_warp.image.array
        )
        np.testing.assert_array_equal(
            coadd_dict['coadd_mfrac_exp'].image.array,
            mfrac_warp.image.array
        )

        # Handling of NaNs is not done on the output from warp_psf
        # and needs special handling.
        assert (np.nansum(coadd_dict['coadd_psf_exp'].image.array)
                == np.nansum(psf_warp.image.array))

        use = ~np.isnan(psf_warp.image.array)
        np.testing.assert_array_equal(
            coadd_dict['coadd_psf_exp'].image.array[use],
            psf_warp.image.array[use]
        )


@pytest.mark.skipif('CATSIM_DIR' not in os.environ,
                    reason='CATSIM_DIR not in os.environ')
@pytest.mark.parametrize('dither', [False, True])
@pytest.mark.parametrize('rotate', [False, True])
def test_coadds_sat(dither, rotate):
    """
    run trials with stars and make sure we get some SAT
    in the coadd mask
    """
    rng = np.random.RandomState(85)

    coadd_dim = 101
    psf_dim = 51
    band = 'i'
    epochs_per_band = 1

    ntrial = 100

    somesat = False
    for i in range(ntrial):
        sim_data = _make_sim(
            rng=rng, psf_type='gauss', bands=[band],
            epochs_per_band=epochs_per_band,
            coadd_dim=coadd_dim, psf_dim=psf_dim,
            dither=dither, rotate=rotate,
            stars=True,
        )

        exps = sim_data['band_data'][band]

        coadd, exp_info = make_coadd_obs(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
        )

        if False:
            import lsst.afw.display as afw_display

            display = afw_display.getDisplay(backend='ds9')
            display.mtv(coadd.coadd_exp)
            display.scale('log', 'minmax')

        mask = coadd.coadd_exp.mask
        sat_flag = mask.getPlaneBitMask('SAT')
        interp_flag = mask.getPlaneBitMask('INTRP')

        wsat = np.where(mask.array & sat_flag != 0)
        wint = np.where(mask.array & interp_flag != 0)

        if wsat[0].size > 0:
            assert wint[0].size > 0, 'expected sat to be interpolated'
            assert np.all(mask.array[wsat] & interp_flag != 0)
            somesat = True
            break

    print('i:', i)
    assert somesat


@pytest.mark.parametrize('max_maskfrac', [-1, 1.1])
def test_coadds_bad_max_maskfrac(max_maskfrac):
    rng = np.random.RandomState(55)

    coadd_dim = 101
    psf_dim = 51

    bands = ['i']
    sim_data = _make_sim(
        rng=rng, psf_type='gauss', bands=bands,
        coadd_dim=coadd_dim, psf_dim=psf_dim,
    )

    # coadd each band separately
    exps = sim_data['band_data']['i']

    with pytest.raises(ValueError):
        coadd, exp_info = make_coadd_obs(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,  # no object poisson noise in sims
            max_maskfrac=max_maskfrac,
        )


if __name__ == '__main__':
    test_coadds_mfrac(dither=True, rotate=True)
    # test_coadds_boundary(rotate=True)
    # test_coadds_smoke(False, False)
