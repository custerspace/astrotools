# NOTE : Assumes only Bayer Filters (as does rawkit as well)
import numpy as np
from scipy.ndimage.filters import convolve
from rawkit.raw import Raw

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
        #self._colorfilter = np.array(rawfile.color_filter_array)
        self._colorfilter = [['R', 'G'], ['G', 'B']]


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
            clfilt = np.array(self.color_filter)
            filtrows = len(clfilt[0])
            filtcols = len(clfilt)
            kernels = self.color_kernels
            color_image = np.zeros((3, *rawimage.shape))
            # The components, R, G, B
            for i in range(3):
                color_image[i] = convolve(rawimage, kernels[i])
            self._color_image = color_image

        return self._color_image

    @property
    def grey_image(self):
        return np.sum(self._color_image, axis=0)

    @property
    def color_kernels(self):
        if self._kernels is None:
            self._make_kernels(self.color_filter)
        return self._kernels


    def _make_kernels(self, cfilt):
        Rkernel = self._make_kernel(cfilt, "R")
        Gkernel = self._make_kernel(cfilt, "G")
        Bkernel = self._make_kernel(cfilt, "B")
        self._kernels = np.array([Rkernel, Gkernel, Bkernel])

    def _make_kernel(self, cfilt, clr):
        kernel = (cfilt == clr).astype(float)
        kernel /= np.sum(kernel)
        return kernel


