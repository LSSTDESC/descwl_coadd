import numpy as np

import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs
from lsst.meas.algorithms import KernelPsf
from lsst.afw.math import FixedKernel
import lsst.afw.image as afw_image
import lsst.afw.math as afw_math
from lsst.pipe.tasks.coaddInputRecorder import (
    CoaddInputRecorderTask,
    CoaddInputRecorderConfig,
)
from lsst.meas.algorithms import CoaddPsf, CoaddPsfConfig

import coord
import ngmix


class MultiBandCoadds(object):
    """
    Parameters
    ----------
    data: dict
        dict keyed by band.  Each entry is a list of "SEObs", which should have
        image, weight, noise, wcs attributes, as well as get_psf method.  For
        example see the simple sim from descwl_shear_testing
    coadd_wcs: galsim wcs
        wcs for final cuadd
    coadd_dims: (nx, ny)
        Currently doing x first rather than row, col
    """
    def __init__(self, *,
                 data,
                 coadd_wcs,
                 coadd_dims):

        self._data = data
        self._coadd_wcs = make_stack_wcs(coadd_wcs)
        self._coadd_dims = coadd_dims

        self._make_exps()
        self._make_coadds()

    @property
    def bands(self):
        """
        get list of bands
        """
        return [k for k in self._data]

    def get_coadd(self, band=None):
        """
        get a coadd

        Parameters
        ----------
        band: str, optional
            Band for coadd, if None return all band coadd

        Returns
        -------
        Coadd for band
        """
        if band is None:
            return self._coadds['all']
        else:
            return self._coadds[band]

    def _make_exps(self):
        """
        make lsst stack exposures for each image
        """

        exps = []
        byband_exps = {}

        for band in self._data:
            bdata = self._data[band]
            byband_exps[band] = []

            for epoch_ind, se_obs in enumerate(bdata):

                wcs = se_obs.wcs
                crpix = wcs.crpix

                cenx, ceny = crpix

                image = se_obs.image.array
                weight = se_obs.weight.array

                # TODO:  get this at the center of coadd
                psf_image = se_obs.get_psf(cenx, ceny).array

                # TODO:  do this better, could be zeros, not uniform, etc
                w = np.where(weight > 0)
                assert w[0].size > 0
                noise_sigma = np.sqrt(1.0/weight[w[0][0], w[1][0]])

                sy, sx = image.shape

                masked_image = afw_image.MaskedImageF(sx, sy)
                masked_image.image.array[:] = image
                masked_image.variance.array[:] = noise_sigma**2

                # TODO:  look for real mask
                masked_image.mask.array[:] = 0

                exp = afw_image.ExposureF(masked_image)

                exp_psf = make_stack_psf(psf_image)

                exp.setPsf(exp_psf)

                # set single WCS
                stack_wcs = make_stack_wcs(wcs)
                exp.setWcs(stack_wcs)

                exps.append(exp)
                byband_exps[band].append(exp)

        self._exps = exps
        self._byband_exps = byband_exps

    def _make_coadds(self):

        # dict are now ordered since python 3.6
        self._coadds = {}

        self._coadds['all'] = CoaddObs(
            exps=self._exps,
            coadd_wcs=self._coadd_wcs,
            coadd_dims=self._coadd_dims,
        )

        for band in self._byband_exps:
            self._coadds[band] = CoaddObs(
                exps=self._byband_exps[band],
                coadd_wcs=self._coadd_wcs,
                coadd_dims=self._coadd_dims,
            )


class CoaddObs(ngmix.Observation):
    def __init__(self, *,
                 exps,
                 coadd_wcs,
                 coadd_dims):

        self._exps = exps
        self._coadd_wcs = coadd_wcs
        self._coadd_dims = coadd_dims

        self._interp = 'lanczos3'

        self._make_warps()
        self._make_coadd()
        self._finish_init()

    def _make_warps(self):
        """
        make the warp images
        """

        # Setup coadd/warp psf model
        input_recorder_config = CoaddInputRecorderConfig()

        input_recorder = CoaddInputRecorderTask(
            config=input_recorder_config, name="dummy",
        )
        coadd_psf_config = CoaddPsfConfig()
        coadd_psf_config.warpingKernelName = self._interp

        # warp stack images to coadd wcs
        warp_config = afw_math.Warper.ConfigClass()

        # currently allows up to lanczos5, but a small change would allow
        # higher order
        warp_config.warpingKernelName = self._interp
        warper = afw_math.Warper.fromConfig(warp_config)

        nx, ny = self._coadd_dims
        sky_box = geom.Box2I(
            geom.Point2I(0, 0),
            geom.Point2I(nx-1, ny-1),
        )

        wexps = []
        weight_list = []
        for i, exp in enumerate(self._exps):

            # Compute variance weight
            stats_ctrl = afw_math.StatisticsControl()
            stat_obj = afw_math.makeStatistics(
                exp.variance,
                exp.mask,
                afw_math.MEANCLIP,
                stats_ctrl,
            )

            mean_var, mean_var_err = stat_obj.getResult(afw_math.MEANCLIP)
            weight = 1.0 / float(mean_var)
            weight_list.append(weight)

            wexp = warper.warpExposure(
                self._coadd_wcs,
                exp,
                maxBBox=exp.getBBox(),
                destBBox=sky_box,
            )

            # Need coadd psf because psf may not be valid over the whole image
            ir_warp = input_recorder.makeCoaddTempExpRecorder(i, 1)
            good_pixel = np.sum(np.isfinite(wexp.image.array))
            ir_warp.addCalExp(exp, i, good_pixel)

            warp_psf = CoaddPsf(
                ir_warp.coaddInputs.ccds,
                self._coadd_wcs,
                coadd_psf_config.makeControl(),
            )
            wexp.getInfo().setCoaddInputs(ir_warp.coaddInputs)
            wexp.setPsf(warp_psf)
            wexps.append(wexp)

        self._wexps = wexps
        self._weights = weight_list
        self._input_recorder = input_recorder
        self._coadd_psf_config = coadd_psf_config

    def _make_coadd(self):
        """
        make the coadd from warp images, as well as psf coadd
        """

        input_recorder = self._input_recorder

        # combine stack images using mean
        stats_flags = afw_math.stringToStatisticsProperty("MEAN")
        stats_ctrl = afw_math.StatisticsControl()

        masked_images = [w.getMaskedImage() for w in self._wexps]
        stacked_image = afw_math.statisticsStack(
            masked_images, stats_flags, stats_ctrl, self._weights, 0, 0)

        stacked_exp = afw_image.ExposureF(stacked_image)
        stacked_exp.getInfo().setCoaddInputs(input_recorder.makeCoaddInputs())
        coadd_inputs = stacked_exp.getInfo().getCoaddInputs()

        # Build coadd psf
        for wexp, weight in zip(self._wexps, self._weights):
            input_recorder.addVisitToCoadd(coadd_inputs, wexp, weight)

        coadd_psf = CoaddPsf(
            coadd_inputs.ccds,
            self._coadd_wcs,
            self._coadd_psf_config.makeControl(),
        )
        stacked_exp.setPsf(coadd_psf)

        self.coadd_exp = stacked_exp

    def _finish_init(self):
        pass
        """
        psf_obs = ngmix.Observation(
            image=psf_image,
            weight=psf_weight,
            jacobian=psf_jac,
        )

        super().__init__(
            image=image,
            noise=noise,
            weight=weight,
            bmask=np.zeros(image.shape, dtype='i4'),
            jacobian=jac,
            psf=psf_obs,
            store_pixels=False,
        )
        """


def make_stack_psf(psf_image):
    """
    make fixed image psf for stack usage
    """
    return KernelPsf(
        FixedKernel(
            afw_image.ImageD(psf_image.astype(np.float))
        )
    )


def make_stack_wcs(wcs):
    """
    convert galsim tan wcs to stack wcs
    """
    crpix = wcs.crpix
    stack_crpix = geom.Point2D(crpix[0], crpix[1])
    cd_matrix = wcs.cd

    crval = geom.SpherePoint(
        wcs.center.ra/coord.radians,
        wcs.center.dec/coord.radians,
        geom.radians,
    )
    return makeSkyWcs(
        crpix=stack_crpix,
        crval=crval,
        cdMatrix=cd_matrix,
    )
