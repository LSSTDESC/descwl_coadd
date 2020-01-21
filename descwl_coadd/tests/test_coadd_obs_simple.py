import numpy as np
import galsim
import pytest

from descwl_shear_sims import SEObs
from ..coadd_simple import CoaddObsSimple

DIMS = (11, 13)


@pytest.fixture
def se_data():

    def psf_function(*, x, y, center_psf=True):
        return galsim.ImageD(np.ones(DIMS) * 6)

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
    data = [SEObs(**se_data)]*3

    coadd_obs = CoaddObsSimple(data)  # noqa
