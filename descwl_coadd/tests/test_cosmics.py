import numpy as np

from descwl_shear_sims import Sim
from ..coadd import MultiBandCoadds


def get_cosmic_flag():
    import lsst.afw.image as afw_image

    mask = afw_image.Mask()
    return 2**mask.getMaskPlane('CR')


def test_cosmics():

    cosmic_flag = get_cosmic_flag()

    rng = np.random.RandomState(8312)
    sim = Sim(
        rng=rng,
        epochs_per_band=2,
        cosmic_rays=True,
    )
    data = sim.gen_sim()

    # coadding individual bands as well as over bands
    coadd_dims = (sim.coadd_dim, )*2
    coadds = MultiBandCoadds(
        data=data,
        coadd_wcs=sim.coadd_wcs,
        coadd_dims=coadd_dims,
    )

    for exp in coadds._exps:
        bmask = exp.mask.array

        w = np.where((bmask & cosmic_flag) != 0)
        print('found:', w[0].size, 'in image')
        assert w[0].size != 0

    for nexp in coadds._noise_exps:
        bmask = nexp.mask.array

        w = np.where((bmask & cosmic_flag) != 0)
        print('found:', w[0].size, 'in noise')
        # currently just make sure there are not too many
        # TODO: tune the finder so it finds none?
        assert w[0].size < 5
