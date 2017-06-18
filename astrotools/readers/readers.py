# NOTE : Assumes only Bayer Filters (as does rawkit as well)
import numpy as np
from scipy.ndimage.filters import convolve
from rawkit.raw import Raw
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

class AstroImage:
    ''' This class is meant to open multiple image formats.
        Currently only works with CR2.
    '''
    def __init__(self, filename):
        self.filename = filename
        # check if file exists, but don't load until needed
        self._rawimage = None
        self._image = None
        self._colorfilter = None
        self._color_image = None
        self._kernels = None

    def _init_raw(self):
        rawfile = Raw(self.filename)
        self._rawimage = np.array(rawfile.raw_image())
        # TODO maybe find better way to get color filter array
        # through a lookup table?
        #self._colorfilter = np.array(rawfile.color_filter_array)
        self._colorfilter = np.array([['R', 'G'], ['G', 'B']])


    @property
    def rawimage(self):
        if self._rawimage is None:
            self._init_raw()
        return self._rawimage

    @property
    def color_filter(self):
        if self._colorfilter is None:
            self._init_raw()
        return self._colorfilter

    @property
    def image(self):
        ''' Create RGB image from Bayer filter.'''
        if self._color_image is None:
            rawimage = self.rawimage
            color_image = _rawbayer2image(rawimage, self.color_filter)
            self._color_image = color_image

        return self._color_image

    @property
    def grey_image(self):
        return np.sum(self._color_image, axis=0)



def _rawbayer2image(img, bayer_mat):
    ''' This converts a raw image to a color image
        assuming the bayer matrix "bayer_mat".

        Parameters
        ----------

        img : the raw image
        bayer_mat : a Bayer matrix of strings "R", "G" or "B"

        Returns
        -------

        cimg : np.ndarray, np.uint8
            The resultant interpolated color image

        Notes
        -----

        This needs to be optimized, and eventually written in C
    '''
    bayer_dims = bayer_mat.shape

    clrs = ["R", "G", "B"]

    # of the original image
    imgy = np.arange(img.shape[0])
    imgx = np.arange(img.shape[1])
    # change to a grid (so there is one x and y per point on img)
    imgY, imgX = np.meshgrid(imgy,imgx,indexing='ij')

    # allocate space for the final color image
    # TODO : find bit dept of image
    cimg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint16)

    # go through each color
    for i, clr in enumerate(clrs):
        subimg = cimg[:,:,i]
        # find the row/cols for the color
        wy, wx = np.where(bayer_mat == clr)
        # if there is only 1 per bayer matrix, then we interpolate
        # using a fast linear interpolation scheme
        if len(wx) == 1:
            wy, wx = wy[0], wx[0]
            values = img[wy::bayer_dims[0], wx::bayer_dims[1]]
            # these are just 1D
            ypts = imgy[wy::bayer_dims[0]]
            xpts = imgx[wx::bayer_dims[1]]
            # create the interpolator object
            rinterp = RegularGridInterpolator((ypts, xpts), values,
                                              bounds_error=False,
                                              fill_value=None)

            boolmat = (imgY % bayer_dims[0] == wy) * \
                        (imgX % bayer_dims[1] == wx)
            # fill in the data
            subimg[boolmat] = img[boolmat]
            # points of missing data
            nboolmat = ~boolmat
            ypts = imgY[nboolmat].ravel()
            xpts = imgX[nboolmat].ravel()
            # for each element in the Bayer matrix
            pts = np.concatenate((ypts[:, np.newaxis],
                                 xpts[:, np.newaxis],
                                 ),axis=1)
            subimg[ypts, xpts] = rinterp(pts)

        # if it's greater than one, then we need a more complex interpolation
        # scheme
        elif len(wx) > 1:
            # more complex interpolation
            boolmat = np.zeros_like(subimg, dtype=bool)
            # find the good points
            for wxi, wyi in zip(wx, wy):
                boolmat += (imgY % bayer_dims[0] == wyi) * \
                            (imgX % bayer_dims[1] == wxi)

            subimg[boolmat] = img[boolmat]

            pts = np.concatenate((imgY[boolmat].ravel()[:,np.newaxis],
                                  imgX[boolmat].ravel()[:,np.newaxis]),axis=1)
            vals = img[boolmat].ravel()
            rinterp = NearestNDInterpolator(pts, vals)

            nboolmat = ~boolmat
            missingpts = np.concatenate((imgY[nboolmat].ravel()[:,np.newaxis],
                                         imgX[nboolmat].ravel()[:,np.newaxis]),axis=1)
            subimg[nboolmat] = rinterp(missingpts)
    return cimg
