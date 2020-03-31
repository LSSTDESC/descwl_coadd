"""
coadd lsst dm stack exposures
"""
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
import lsst.log

import coord
import ngmix

from . import vis
from .interp import interpolate_image_and_noise

EDGE = afw_image.Mask.getPlaneBitMask('EDGE')

# we interpolate the saturated pixels but not full star
# or bleed masks
FLAGS2INTERP = (
    afw_image.Mask.getPlaneBitMask('BAD') |
    afw_image.Mask.getPlaneBitMask('CR') |
    afw_image.Mask.getPlaneBitMask('SAT')
)


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
                 psf_dims,
                 byband=True,
                 use_stack_interp=False,
                 show=False,
                 loglevel='info'):

        self._show = show
        self.log = lsst.log.getLogger("MultiBandCoadds")
        self.log.setLevel(getattr(lsst.log, loglevel.upper()))
        self.loglevel = loglevel

        self.data = data
        self.coadd_wcs = coadd_wcs
        self.coadd_dims = coadd_dims
        self.psf_dims = psf_dims
        self.byband = byband
        self.use_stack_interp = use_stack_interp

        self._make_exps()
        self._make_coadds()

    @property
    def bands(self):
        """
        get list of bands
        """
        return [k for k in self.data]

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

        self.log.info('making exps')

        cwcs = self.coadd_wcs

        exps = []
        noise_exps = []
        psf_exps = []
        byband_exps = {}
        byband_noise_exps = {}
        byband_psf_exps = {}

        for band in self.data:
            bdata = self.data[band]
            byband_exps[band] = []
            byband_noise_exps[band] = []
            byband_psf_exps[band] = []

            for epoch_ind, se_obs in enumerate(bdata):

                wcs = se_obs.wcs
                pos = wcs.toImage(cwcs.center)

                image = se_obs.image.array
                noise = se_obs.noise.array
                bmask = se_obs.bmask.array
                weight = se_obs.weight.array

                if not self.use_stack_interp:
                    zero_bits(image=image, noise=noise, mask=bmask, flags=EDGE)
                    image, noise = interpolate_image_and_noise(
                        image=image,
                        noise=noise,
                        weight=weight,
                        bmask=bmask,
                        bad_flags=FLAGS2INTERP,
                    )
                    if self._show:
                        vis.show_images([
                            image,
                            bmask,
                            noise,
                        ])

                weight = se_obs.weight.array

                psf_gsimage, psf_offset = se_obs.get_psf(
                    pos.x,
                    pos.y,
                    center_psf=False,
                    get_offset=True,
                )
                psf_image = psf_gsimage.array

                # TODO: deal with zeros
                w = np.where(weight > 0)
                assert w[0].size == weight.size
                noise_var = 1.0/weight

                sy, sx = image.shape

                masked_image = afw_image.MaskedImageF(sx, sy)
                masked_image.image.array[:, :] = image
                masked_image.variance.array[:, :] = noise_var
                masked_image.mask.array[:, :] = bmask

                nmasked_image = afw_image.MaskedImageF(sx, sy)
                nmasked_image.image.array[:, :] = noise
                nmasked_image.variance.array[:, :] = noise_var
                nmasked_image.mask.array[:, :] = bmask

                # an exp for the PSF
                # we need to put a var here that is consistent with the
                # main image to get a consistent coadd.  I'm not sure
                # what the best choice is, using median for now TODO
                pny, pnx = psf_image.shape
                pmasked_image = afw_image.MaskedImageF(pny, pnx)
                pmasked_image.image.array[:, :] = psf_image
                pmasked_image.variance.array[:, :] = np.median(noise_var)
                pmasked_image.mask.array[:, :] = 0

                exp = afw_image.ExposureF(masked_image)
                nexp = afw_image.ExposureF(nmasked_image)
                psf_exp = afw_image.ExposureF(pmasked_image)

                exp.setPsf(make_stack_psf(psf_image))
                nexp.setPsf(make_stack_psf(psf_image))

                exp.setWcs(make_stack_wcs(wcs))
                nexp.setWcs(make_stack_wcs(wcs))
                psf_exp.setWcs(make_stack_psf_wcs(
                    dims=psf_gsimage.array.shape,
                    jac=psf_gsimage.wcs,
                    offset=psf_offset,
                    world_origin=wcs.center,
                ))

                detector = DetectorWrapper().detector
                exp.setDetector(detector)
                nexp.setDetector(detector)
                psf_exp.setDetector(detector)

                if self.use_stack_interp:
                    add_cosmics_to_noise(exp=exp, noise_exp=nexp)
                    add_badcols_to_noise(exp=exp, noise_exp=nexp)

                    repair_exp(exp, show=False)
                    repair_exp(nexp, show=False)

                if self._show:
                    vis.show_image_and_mask(exp)

                exps.append(exp)
                noise_exps.append(nexp)
                byband_exps[band].append(exp)
                byband_noise_exps[band].append(nexp)

                psf_exps.append(psf_exp)
                byband_psf_exps[band].append(psf_exp)

        self.exps = exps
        self.noise_exps = noise_exps
        self.psf_exps = psf_exps
        self.byband_exps = byband_exps
        self.byband_noise_exps = byband_noise_exps
        self.byband_psf_exps = byband_psf_exps

    def _make_coadds(self):
        """
        make all coadds
        """
        self.log.info('making coadds')

        # dict are now ordered since python 3.6
        self.coadds = {}

        if self.byband:
            for band in self.byband_exps:
                self.coadds[band] = CoaddObs(
                    exps=self.byband_exps[band],
                    psf_exps=self.byband_psf_exps[band],
                    noise_exps=self.byband_noise_exps[band],
                    coadd_wcs=self.coadd_wcs,
                    coadd_dims=self.coadd_dims,
                    psf_dims=self.psf_dims,
                )

        self.coadds['all'] = CoaddObs(
            exps=self.exps,
            noise_exps=self.noise_exps,
            psf_exps=self.psf_exps,
            coadd_wcs=self.coadd_wcs,
            coadd_dims=self.coadd_dims,
            psf_dims=self.psf_dims,
        )
        if self._show:
            self.coadds['all'].show()


class CoaddObs(ngmix.Observation):
    """
    make coadd exposure for the input exposures and noise exposures
    """
    def __init__(self, *,
                 exps,
                 noise_exps,
                 psf_exps,
                 coadd_wcs,
                 coadd_dims,
                 psf_dims):

        import galsim

        self.exps = exps
        self.psf_exps = psf_exps
        self.noise_exps = noise_exps
        self.galsim_wcs = coadd_wcs
        self.coadd_wcs = make_stack_wcs(coadd_wcs)
        self.coadd_dims = coadd_dims
        self.psf_dims = psf_dims

        ceny, cenx = (np.array(coadd_dims)-1)/2
        image_pos = galsim.PositionD(x=cenx, y=ceny)
        jac = coadd_wcs.local(image_pos=image_pos)

        self.coadd_psf_wcs = make_stack_psf_wcs(
            dims=psf_dims,
            offset=galsim.PositionD(x=0, y=0),
            jac=jac,
            world_origin=coadd_wcs.center,
        )

        self.interp = 'lanczos3'

        self._make_coadds()
        self._finish_init()

    def show(self):
        vis.show_images(
            [
                self.image,
                self.coadd_exp.mask.array,
                self.noise,
                self.coadd_noise_exp.mask.array,
                self.psf.image,
                # self.weight,
            ],
        )

    def _make_coadds(self):
        """
        make warps and coadds for images and noise fields
        """
        psf_data = self._make_warps(
            exps=self.psf_exps,
            dims=self.psf_dims,
            wcs=self.coadd_psf_wcs,
            dopsf=False,
        )
        coadd_psf_exp = self._make_coadd(**psf_data)
        pimage = coadd_psf_exp.image.array
        wbad = np.where(~np.isfinite(pimage))

        if wbad[0].size == pimage.size:
            raise ValueError('no good pixels in the psf')

        pimage[wbad] = 0.0

        # vis.show_2images(self.psf_exps[0].image.array, pimage)

        image_data = self._make_warps(
            exps=self.exps,
            dims=self.coadd_dims,
            wcs=self.coadd_wcs,
            dopsf=True,
        )
        self.coadd_exp = self._make_coadd(**image_data)
        self.coadd_exp.setPsf(make_stack_psf(pimage))

        noise_data = self._make_warps(
            exps=self.noise_exps,
            dims=self.coadd_dims,
            wcs=self.coadd_wcs,
            dopsf=True,
        )
        self.coadd_noise_exp = self._make_coadd(**noise_data)
        self.coadd_noise_exp.setPsf(make_stack_psf(pimage))

    def _make_warps(self, *, exps, dims, wcs, dopsf=False):
        """
        make the warp images
        """

        # Setup coadd/warp psf model
        input_recorder_config = CoaddInputRecorderConfig()

        input_recorder = CoaddInputRecorderTask(
            config=input_recorder_config, name="dummy",
        )

        if dopsf:
            coadd_psf_config = CoaddPsfConfig()
            coadd_psf_config.warpingKernelName = self.interp

        # warp stack images to coadd wcs
        warp_config = afw_math.Warper.ConfigClass()

        # currently allows up to lanczos5, but a small change would allow
        # higher order
        warp_config.warpingKernelName = self.interp
        warper = afw_math.Warper.fromConfig(warp_config)

        nx, ny = dims
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
                wcs,
                exp,
                maxBBox=exp.getBBox(),
                destBBox=sky_box,
            )

            # Need coadd psf because psf may not be valid over the whole image
            ir_warp = input_recorder.makeCoaddTempExpRecorder(i, 1)
            good_pixel = np.sum(np.isfinite(wexp.image.array))
            ir_warp.addCalExp(exp, i, good_pixel)

            if dopsf:
                warp_psf = CoaddPsf(
                    ir_warp.coaddInputs.ccds,
                    wcs,
                    coadd_psf_config.makeControl(),
                )

            wexp.getInfo().setCoaddInputs(ir_warp.coaddInputs)

            if dopsf:
                wexp.setPsf(warp_psf)

            wexps.append(wexp)

        data = {
            'wexps': wexps,
            'weights': weight_list,
            'input_recorder': input_recorder,
        }
        if dopsf:
            data['psf_config'] = coadd_psf_config

        return data

    def _make_coadd(self, *, wexps, weights, input_recorder, psf_config=None):
        """
        make a coadd from warp images, as well as psf coadd
        """

        # combine stack images using mean
        stats_flags = afw_math.stringToStatisticsProperty("MEAN")
        stats_ctrl = afw_math.StatisticsControl()
        stats_ctrl.setCalcErrorFromInputVariance(True)
        badmask = afw_image.Mask.getPlaneBitMask(['EDGE'])
        stats_ctrl.setAndMask(badmask)

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

        if psf_config is not None:
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
            ormask=self.coadd_exp.mask.array,
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


def make_stack_psf_wcs(*, dims, offset, jac, world_origin):
    """
    convert the galsim jacobian wcs to stack wcs
    for a tan projection

    Parameters
    ----------
    dims: (ny, nx)
        dims of the psf
    offset: seq or array
        xoffset, yoffset
    jac: galsim jacobian
        From wcs
    world_origin: origin of wcs
        get from coadd_wcs.center
    """
    import galsim

    cy, cx = (np.array(dims)-1)/2
    cy += offset.y
    cx += offset.x
    origin = galsim.PositionD(x=cx, y=cy)

    tan_wcs = galsim.TanWCS(
        affine=galsim.AffineTransform(
            jac.dudx, jac.dudy, jac.dvdx, jac.dvdy,
            origin=origin,
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )

    return make_stack_wcs(tan_wcs)


def repair_exp(exp, show=False, border_size=None):
    """
    run a RepairTask, currently not sending defects just
    finding cosmics and interpolating them

    Parameters
    ----------
    exp:
        an Exposure from the stack
    border_size: bool
        border size to zero
    show: bool
        If True, show the image
    """
    from lsst.pipe.tasks.repair import RepairTask, RepairConfig

    if show:
        print('before repair')
        vis.show_image(exp.image.array)

    if border_size is not None:
        b = border_size
        ny, nx = exp.image.array.shape
        exp.image.array[0:b, :] = 0
        exp.image.array[ny-b:, :] = 0
        exp.image.array[:, 0:b] = 0
        exp.image.array[:, nx-b:] = 0

    repair_config = RepairConfig()
    repair_task = RepairTask(config=repair_config)
    repair_task.run(exposure=exp)

    if show:
        print('after repair')
        vis.show_image(exp.image.array)


def add_cosmics_to_noise(*, exp, noise_exp, value=1.0e18):
    """
    add fake cosmics to the noise exposure wherever
    they are set in the real image

    TODO: get a realistic value for the real data
    """

    CR = 2**afw_image.Mask.getMaskPlane('CR')  # noqa
    w = np.where((exp.mask.array & CR) != 0)
    print('pixels for cosmics:', w[0].size)
    if w[0].size > 0:
        noise_exp.image.array[w] = value


def add_badcols_to_noise(*, exp, noise_exp):
    """
    Set bad cols in the noise image based on the real
    image bits
    """

    BAD = 2**afw_image.Mask.getMaskPlane('BAD')  # noqa
    w = np.where((exp.mask.array & BAD) != 0)
    print('pixels for badcols:', w[0].size)
    if w[0].size > 0:
        noise_exp.image.array[w] = exp.image.array[w]


def zero_bits(*, image, noise, mask, flags):
    w = np.where((mask & flags) != 0)
    # w = np.where(mask != 0)
    if w[0].size > 0:
        image[w] = 0.0
        noise[w] = 0.0
