import os.path
import numpy as np
#from astrotools.readers.readers import AstroImage

PDIR = os.path.expanduser("~/pictures/17-03-18-custer-whirlpool")
prefix = "IMG_"
suffix = ".CR2"
whirlpool_nums = np.arange(3432, 3530+1)
darks = np.arange(3535, 3555+1)

filename = PDIR + "/" + prefix + "3772"+suffix

#filename = "/home/julienl/astropictures/170603-nick/IMG_9655.CR2"

from rawkit.raw import Raw
im = Raw(filename)
raw = im.raw_image()
print(raw)


