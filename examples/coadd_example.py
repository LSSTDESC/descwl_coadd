
#!/usr/bin/env python
import galsim
import numpy as np
import pylab
import coord

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.afw.math as afwMath

from lsst.meas.base import NoiseReplacerConfig, NoiseReplacer
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.afw.geom import makeSkyWcs, makeCdMatrix
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.algorithms import SingleGaussianPsf
from lsst.meas.algorithms import WarpedPsf
from lsst.pipe.tasks.coaddInputRecorder import CoaddInputRecorderTask, CoaddInputRecorderConfig
from lsst.meas.algorithms import CoaddPsf, CoaddPsfConfig

from astropy.visualization import ZScaleInterval


class powerLaw:

    def __init__(self, min, max, gamma, rng):
        self.min = min
        self.max = max
        self.gamma_p1 = gamma+1
        self.rng = rng
        if self.gamma_p1 == 0:
            self.base = np.log(self.min)
            self.norm = np.log(self.max/self.min)
        else:
            self.base = np.power(self.min, self.gamma_p1)
            self.norm = np.power(self.max, self.gamma_p1) - self.base

    def sample(self):
        v = self.rng() * self.norm + self.base
        if self.gamma_p1 == 0:
            return np.exp(v)
        else:
            return np.power(v, 1./self.gamma_p1)


zscale = ZScaleInterval()
seed = 12345
rng = galsim.UniformDeviate(seed)
gauss = galsim.GaussianDeviate(rng)
np.random.seed(seed)

# The number of sub images to simulate
n_image = 15

# The number of galaxies
ngal = 100

# size of each sub image
image_sizey = 400
image_sizex = 400

# size of full image
tot_image_sizey = 2*image_sizey
tot_image_sizex = 2*image_sizex

# size of postage stamp
stamp_size = 64

scale = 1.0

psf_size = 1.6
noise_sigma = 0.06

flux_pl = powerLaw(60, 100, -0.75, rng)
size_pl = powerLaw(2, 3.5, -1, rng)


apply_shear = True
sigma_e = 0.8

# Setup galsim sky wcs for coadd
tot_origin = galsim.PositionD(tot_image_sizex/2, tot_image_sizey/2)
tot_world_origin = galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees)
affine = galsim.AffineTransform(scale, 0, 0, scale, origin=tot_origin)
tot_wcs = galsim.TanWCS(affine, world_origin=tot_world_origin)

min_ra = tot_wcs.toWorld(galsim.PositionD(0, 0)).ra/coord.arcmin
min_dec = tot_wcs.toWorld(galsim.PositionD(0, 0)).dec/coord.arcmin
max_ra = tot_wcs.toWorld(galsim.PositionD(
    tot_image_sizex, tot_image_sizey)).ra/coord.arcmin
max_dec = tot_wcs.toWorld(galsim.PositionD(
    tot_image_sizex, tot_image_sizey)).dec/coord.arcmin

noise = galsim.GaussianNoise(rng, sigma=noise_sigma)
psf = galsim.Gaussian(sigma=psf_size)
psf_bounds = galsim.BoundsI(0, stamp_size-1, 0, stamp_size-1)
psf_image = galsim.ImageF(psf_bounds, scale=scale)
psf.drawImage(image=psf_image, scale=scale)

sbounds = galsim.BoundsI(0, image_sizex-1, 0, image_sizey-1)
tot_bounds = galsim.BoundsI(0, tot_image_sizex-1, 0, tot_image_sizey-1)

# generate the images with each image offset and rotated from central point
gwcs = []
images = []
thetas = []
for n in range(n_image):
    xpos = np.random.rand()*(tot_image_sizex - image_sizex) + image_sizex/2
    ypos = np.random.rand()*(tot_image_sizey - image_sizey) + image_sizey/2
    c = tot_wcs.toWorld(galsim.PositionD(xpos, ypos))
    galsim_origin = galsim.PositionD(image_sizex/2, image_sizey/2)
    galsim_world_origin = galsim.CelestialCoord(c.ra, c.dec)

    theta = np.radians(np.random.rand()*360)
    thetas.append(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    vec = np.array([[scale, 0], [0, scale]])

    cd = np.dot(R, vec)
    affine = galsim.AffineTransform(
        cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1], origin=galsim_origin)
    galsim_wcs = galsim.TanWCS(affine, world_origin=galsim_world_origin)
    gwcs.append(galsim_wcs)
    image = galsim.ImageF(sbounds, wcs=galsim_wcs)
    images.append(image)


# generate galaxies
for igal in range(ngal):

    ra = np.random.rand()*(max_ra - min_ra) + min_ra
    dec = np.random.rand()*(max_dec - min_dec) + min_dec

    exp_e1 = 2
    exp_e2 = 2
    while np.sqrt(exp_e1**2 + exp_e2**2) > sigma_e:
        exp_e1 = gauss()*sigma_e
        exp_e2 = gauss()*sigma_e

    exp_radius = size_pl.sample()
    flux = flux_pl.sample()
    disk = galsim.Exponential(half_light_radius=exp_radius*scale, flux=flux)
    disk = disk.shear(e1=exp_e1, e2=exp_e2)

    cgal = galsim.Convolve([disk, psf])

    for wcs, image in zip(gwcs, images):
        c = galsim.CelestialCoord(ra*coord.arcmin, dec*coord.arcmin)
        pos = wcs.toImage(c)
        if image.bounds.includes(pos) is False:
            continue

        tmp = cgal.drawImage(image=image, add_to_image=True,
                             offset=(pos.x-image_sizex/2, pos.y-image_sizey/2))

for image in images:
    image.addNoise(noise)

# Generate warped images with galsim to compare
full_images = []
interp = 'lanczos3'
for image in images:
    fimage = galsim.ImageF(tot_bounds, wcs=tot_wcs)
    int_im = galsim.InterpolatedImage(image, wcs=image.wcs, x_interpolant=interp)
    pos = tot_wcs.toImage(image.wcs.center)
    int_im.drawImage(fimage, method='no_pixel', wcs=tot_wcs,
                     offset=(pos.x-tot_image_sizex/2, pos.y-tot_image_sizey/2))
    full_images.append(fimage)

# Convert galsim images to DM
exp_psf = SingleGaussianPsf(stamp_size, stamp_size, psf_size)

exps = []
for image, theta in zip(images, thetas):
    masked_image = afwImage.MaskedImageF(image_sizex, image_sizey)
    masked_image.image.array[:] = image.array
    masked_image.variance.array[:] = noise_sigma**2
    masked_image.mask.array[:] = 0
    exp = afwImage.ExposureF(masked_image)

    exp.setPsf(exp_psf)

    # set single WCS
    orientation = theta*geom.radians
    cd_matrix = makeCdMatrix(
        scale=scale*geom.arcseconds, orientation=orientation)
    crpix = geom.Point2D(image_sizex/2, image_sizey/2)
    crval = geom.SpherePoint(image.wcs.center.ra/coord.radians,
                             image.wcs.center.dec/coord.radians,
                             geom.radians)
    wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
    exp.setWcs(wcs)
    exps.append(exp)

# setup stack coadd wcs
global_cd_matrix = makeCdMatrix(scale=scale*geom.arcseconds)
crpix = geom.Point2D(tot_image_sizex/2, tot_image_sizey/2)
crval = geom.SpherePoint(0, 0, geom.degrees)
global_wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=global_cd_matrix)
sky_box = geom.Box2I(geom.Point2I(0, 0), geom.Point2I(
    tot_image_sizex-1, tot_image_sizey-1))

# Setup coadd/warp psf model
input_recorder_config = CoaddInputRecorderConfig()
input_recorder = CoaddInputRecorderTask(
    config=input_recorder_config, name="dummy")
coadd_psf_config = CoaddPsfConfig()
coadd_psf_config.warpingKernelName = 'lanczos3'

# warp stack images to coadd wcs
warp_config = afwMath.Warper.ConfigClass()
# currently allows up to lanczos5, but a small change would allow higher order
warp_config.warpingKernelName = 'lanczos3'
warper = afwMath.Warper.fromConfig(warp_config)

wexps = []
weight_list = []
for i, exp in enumerate(exps):

    # Compute variance weight
    stats_ctrl = afwMath.StatisticsControl()
    stat_obj = afwMath.makeStatistics(exp.variance, exp.mask,
                                      afwMath.MEANCLIP, stats_ctrl)
    mean_var, mean_var_err = stat_obj.getResult(afwMath.MEANCLIP)
    weight = 1.0 / float(mean_var)
    weight_list.append(weight)

    wexp = warper.warpExposure(
        global_wcs, exp, maxBBox=exp.getBBox(), destBBox=sky_box)

    # Need coadd psf because psf may not be valid over the whole image
    ir_warp = input_recorder.makeCoaddTempExpRecorder(i, 1)
    good_pixel = np.sum(np.isfinite(wexp.image.array))
    ir_warp.addCalExp(exp, i, good_pixel)

    warp_psf = CoaddPsf(ir_warp.coaddInputs.ccds, global_wcs,
                        coadd_psf_config.makeControl())
    wexp.getInfo().setCoaddInputs(ir_warp.coaddInputs)
    wexp.setPsf(warp_psf)
    wexps.append(wexp)

# combine stack images using mean
stats_flags = afwMath.stringToStatisticsProperty("MEAN")
stats_ctrl = afwMath.StatisticsControl()

masked_images = [w.getMaskedImage() for w in wexps]
stacked_image = afwMath.statisticsStack(
    masked_images, stats_flags, stats_ctrl, weight_list, 0, 0)

stacked_exp = afwImage.ExposureF(stacked_image)
stacked_exp.getInfo().setCoaddInputs(input_recorder.makeCoaddInputs())
coadd_inputs = stacked_exp.getInfo().getCoaddInputs()

# Build coadd psf
for wexp, weight in zip(wexps, weight_list):
    input_recorder.addVisitToCoadd(coadd_inputs, wexp, weight)

coadd_psf = CoaddPsf(coadd_inputs.ccds, global_wcs,
                     coadd_psf_config.makeControl())
stacked_exp.setPsf(coadd_psf)