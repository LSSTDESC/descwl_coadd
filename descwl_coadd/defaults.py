# Interpolation kernel for warping
DEFAULT_INTERP = 'lanczos3'

# Areas in the image with these flags set will get interpolated.

FLAGS2INTERP = ('BAD', 'CR', 'SAT')

# A name for the mask plane to indicate boundary pixels.
BOUNDARY_BIT_NAME = 'BOUNDARY'

# Width of boundary used for checking if an exposure has an edge
# in the coadd region. Use 3 to match lanczos3 kernel size
BOUNDARY_SIZE = 3

# If an exposure has more than this fraction of its pixels masked it is not
# included in the coadd
# TODO make configurable
MAX_MASKFRAC = 0.9

# used for getting good pixels around a set of bad pixels
DEFAULT_GOOD_PIXEL_BUFF = 4
