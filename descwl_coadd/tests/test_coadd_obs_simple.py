import numpy as np
import galsim
import pytest

from descwl_shear_sims.se_obs import SEObs
from ..coadd_simple import MultiBandCoaddsSimple

DIMS = (11, 13)


@pytest.fixture
def se_data():

    def psf_function(*, x, y, center_psf=True, get_offset=False):
        im = galsim.ImageD(np.ones(DIMS) * 6)
        if get_offset:
            offset = galsim.PositionD(x=0, y=0)
            return im, offset
        else:
            return im

    data = {
        'image': galsim.ImageD(np.ones(DIMS)),
        'weight': galsim.ImageD(np.ones(DIMS) * 2),
        'noise': galsim.ImageD(np.ones(DIMS) * 3),
        'bmask': galsim.ImageI(np.ones(DIMS) * 4),
        'ormask': galsim.ImageI(np.ones(DIMS) * 5),
        'wcs': galsim.PixelScale(0.2),
        'psf_function': psf_function,
    }
    return data


def test_coadd_obs_simple_smoke(se_data):
    data = {
        'r': [SEObs(**se_data)]*3,
        'i': [SEObs(**se_data)]*3,
        'z': [SEObs(**se_data)]*3,
    }

    MultiBandCoaddsSimple(data=data)
