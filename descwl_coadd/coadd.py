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
    byband: bool
        If True, make coadds for individual bands as well as over all
        bands
    """
    def __init__(self, *,
                 data,
                 coadd_wcs,
                 coadd_dims,
                 byband=True):

        self._data = data
        self._coadd_wcs = coadd_wcs
        self._coadd_dims = coadd_dims
        self._byband = byband

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
            Band for coadd, if None return coadd over all bands

        Returns
        -------
        Coadd for band
        """
        if band is None:
            return self.coadds['all']
        else:
            return self.coadds[band]

    def _make_exps(self):
        """
        make lsst stack exposures for each image and noise image
        """

        from lsst.afw.cameraGeom.testUtils import DetectorWrapper

        cwcs = self._coadd_wcs

        exps = []
        noise_exps = []
        byband_exps = {}
        byband_noise_exps = {}

        for band in self._data:
            bdata = self._data[band]
            byband_exps[band] = []
            byband_noise_exps[band] = []

            for epoch_ind, se_obs in enumerate(bdata):

                wcs = se_obs.wcs
                pos = wcs.toImage(cwcs.center)

                image = se_obs.image.array
                noise = se_obs.noise.array
                weight = se_obs.weight.array

                psf_image = se_obs.get_psf(
                    pos.x,
                    pos.y,
                    center_psf=True,
                ).array

                # TODO: deal with zeros
                w = np.where(weight > 0)
                assert w[0].size == weight.size
                noise_var = 1.0/weight

                sy, sx = image.shape

                # TODO:  look for real mask
                masked_image = afw_image.MaskedImageF(sx, sy)
                masked_image.image.array[:, :] = image
                masked_image.variance.array[:, :] = noise_var
                masked_image.mask.array[:, :] = 0

                nmasked_image = afw_image.MaskedImageF(sx, sy)
                nmasked_image.image.array[:, :] = noise
                nmasked_image.variance.array[:, :] = noise_var
                nmasked_image.mask.array[:, :] = 0

                exp = afw_image.ExposureF(masked_image)
                nexp = afw_image.ExposureF(nmasked_image)

                exp.setPsf(make_stack_psf(psf_image))
                nexp.setPsf(make_stack_psf(psf_image))

                exp.setWcs(make_stack_wcs(wcs))
                nexp.setWcs(make_stack_wcs(wcs))

                detector = DetectorWrapper().detector
                exp.setDetector(detector)
                nexp.setDetector(detector)

                repair_exp(exp)
                repair_exp(nexp)

                exps.append(exp)
                noise_exps.append(nexp)
                byband_exps[band].append(exp)
                byband_noise_exps[band].append(nexp)

        self._exps = exps
        self._noise_exps = noise_exps
        self._byband_exps = byband_exps
        self._byband_noise_exps = byband_noise_exps

    def _make_coadds(self):
        """
        make all coadds
        """
        # dict are now ordered since python 3.6
        self.coadds = {}

        if self._byband:
            for band in self._byband_exps:
                self.coadds[band] = CoaddObs(
                    exps=self._byband_exps[band],
                    noise_exps=self._byband_noise_exps[band],
                    coadd_wcs=self._coadd_wcs,
                    coadd_dims=self._coadd_dims,
                )

        self.coadds['all'] = CoaddObs(
            exps=self._exps,
            noise_exps=self._noise_exps,
            coadd_wcs=self._coadd_wcs,
            coadd_dims=self._coadd_dims,
        )


class CoaddObs(ngmix.Observation):
    """
    make coadd exposure for the input exposures and noise exposures
    """
    def __init__(self, *,
                 exps,
                 noise_exps,
                 coadd_wcs,
                 coadd_dims):

        self._exps = exps
        self._noise_exps = noise_exps
        self.galsim_wcs = coadd_wcs
        self.coadd_wcs = make_stack_wcs(coadd_wcs)
        self._coadd_dims = coadd_dims

        self._interp = 'lanczos3'

        self._make_coadds()
        self._finish_init()

    def _make_coadds(self):
        """
        make warps and coadds for images and noise fields
        """
        image_data = self._make_warps(self._exps)
        self.coadd_exp = self._make_coadd(**image_data)

        noise_data = self._make_warps(self._noise_exps)
        self.coadd_noise_exp = self._make_coadd(**noise_data)

    def _make_warps(self, exps):
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
        for i, exp in enumerate(exps):

            # Compute variance weight
            stats_ctrl = afw_math.StatisticsControl()
            stats_ctrl.setCalcErrorFromInputVariance(True)
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
                self.coadd_wcs,
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
                self.coadd_wcs,
                coadd_psf_config.makeControl(),
            )
            wexp.getInfo().setCoaddInputs(ir_warp.coaddInputs)
            wexp.setPsf(warp_psf)
            wexps.append(wexp)

        return {
            'wexps': wexps,
            'weights': weight_list,
            'input_recorder': input_recorder,
            'psf_config': coadd_psf_config,
        }

    def _make_coadd(self, *, wexps, weights, input_recorder, psf_config):
        """
        make a coadd from warp images, as well as psf coadd
        """

        # combine stack images using mean
        stats_flags = afw_math.stringToStatisticsProperty("MEAN")
        stats_ctrl = afw_math.StatisticsControl()
        stats_ctrl.setCalcErrorFromInputVariance(True)

        masked_images = [w.getMaskedImage() for w in wexps]
        stacked_image = afw_math.statisticsStack(
            masked_images, stats_flags, stats_ctrl, weights, 0, 0)

        stacked_exp = afw_image.ExposureF(stacked_image, self.coadd_wcs)
        # stacked_exp.setWcs(self.coadd_wcs)
        stacked_exp.getInfo().setCoaddInputs(input_recorder.makeCoaddInputs())
        coadd_inputs = stacked_exp.getInfo().getCoaddInputs()

        # Build coadd psf
        for wexp, weight in zip(wexps, weights):
            input_recorder.addVisitToCoadd(coadd_inputs, wexp, weight)

        coadd_psf = CoaddPsf(
            coadd_inputs.ccds,
            self.coadd_wcs,
            psf_config.makeControl(),
        )
        stacked_exp.setPsf(coadd_psf)

        return stacked_exp

    def _get_jac(self, *, cenx, ceny):
        """
        get jacobian at the coadd image center, and make
        an ngmix jacobian with center specified (this is not the
        position used to evaluate the jacobian)
        """
        import galsim

        crpix = self.galsim_wcs.crpix
        galsim_pos = galsim.PositionD(x=crpix[0], y=crpix[1])

        galsim_jac = self.galsim_wcs.jacobian(image_pos=galsim_pos)

        return ngmix.Jacobian(
            x=cenx,
            y=ceny,
            dudx=galsim_jac.dudx,
            dudy=galsim_jac.dudy,
            dvdx=galsim_jac.dvdx,
            dvdy=galsim_jac.dvdy,
        )

    def _get_psf_obs(self):
        """
        get the psf observation
        """
        crpix = self.galsim_wcs.crpix
        stack_pos = geom.Point2D(crpix[0], crpix[1])

        psf_obj = self.coadd_exp.getPsf()
        psf_image = psf_obj.computeKernelImage(stack_pos).array

        psf_cen = (np.array(psf_image.shape)-1.0)/2.0

        psf_jac = self._get_jac(cenx=psf_cen[1], ceny=psf_cen[0])

        psf_err = psf_image.max()*0.0001
        psf_weight = psf_image*0 + 1.0/psf_err**2
        return ngmix.Observation(
            image=psf_image,
            weight=psf_weight,
            jacobian=psf_jac,
        )

    def _finish_init(self):
        """
        finish the init by sending the image etc. to the
        Observation init
        """
        psf_obs = self._get_psf_obs()  # noqa

        image = self.coadd_exp.image.array
        noise = self.coadd_noise_exp.image.array

        var = self.coadd_exp.variance.array.copy()
        # print('var:', var)
        # print('image:', image)
        wnf = np.where(~np.isfinite(var))

        if wnf[0].size == image.size:
            raise ValueError('no good variance values')

        if wnf[0].size > 0:
            var[wnf] = -1

        weight = var.copy()
        weight[:, :] = 0.0

        w = np.where(var > 0)
        weight[w] = 1.0/var[w]

        if wnf[0].size > 0:
            # medval = np.sqrt(np.median(var[w]))
            # weight[wbad] = medval
            # TODO: add noise instead based on medval, need to send in rng
            image[wnf] = 0.0
            noise[wnf] = 0.0

        cen = (np.array(image.shape)-1)/2
        jac = self._get_jac(cenx=cen[1], ceny=cen[0])

        super().__init__(
            image=image,
            noise=noise,
            weight=weight,
            bmask=np.zeros(image.shape, dtype='i4'),
            jacobian=jac,
            psf=psf_obs,
            store_pixels=False,
        )


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


def repair_exp(exp):
    """
    run a RepairTask, currently not sending defects just
    finding cosmics and interpolating them
    """
    from lsst.pipe.tasks.repair import RepairTask, RepairConfig
    repair_config = RepairConfig()
    repair_task = RepairTask(config=repair_config)
    repair_task.run(exposure=exp)
