''' 
    Astronomy images shift + register
'''
# spatial correlate and fit vs time
from pylab import *
import os.path
from PIL import Image
from rawkit.raw import Raw
from skbeam.core.utils import bin_edges_to_centers


#PDIR = os.path.expanduser("~/pictures/17-02-26-custerandromedabodes")
PDIR = os.path.expanduser("~/pictures/17-03-18-custer-whirlpool")
prefix = "IMG_"
suffix = ".JPG"
whirlpool_nums = np.arange(3432, 3530+1)
darks = np.arange(3535, 3555+1)

suffix = ".CR2"
prefix = "IMG_"
n = 3787
filename = "{}/{}{}{}".format(PDIR, prefix, n, suffix)


im = Raw(filename)
data = np.array(im.raw_image())
R = data[::2,::2]
G = (data[::2,1::2]+data[1::2,::2])*.5
B = data[1::2,1::2]

Grey = R+G+B

# process each channel
Rhist, Redges = np.histogram(R.ravel(), bins=1000)
Ghist, Gedges = np.histogram(G.ravel(), bins=1000)
Bhist, Bedges = np.histogram(B.ravel(), bins=1000)
Greyhist, Greyedges = np.histogram(Grey.ravel(), bins=1000)

Rcenters = bin_edges_to_centers(Redges)
Gcenters = bin_edges_to_centers(Gedges)
Bcenters = bin_edges_to_centers(Bedges)
Greycenters = bin_edges_to_centers(Greyedges)

figure(1);clf();
subplot(2,2,1)
plot(np.log10(Rcenters), Rhist)
subplot(2,2,2)
plot(np.log10(Gcenters), Ghist)
subplot(2,2,3)
plot(np.log10(Bcenters), Bhist)
subplot(2,2,4)
plot(np.log10(Greycenters), Greyhist)

# now fit each
from tools.Fitters1D import Gauss1DFitter
gfit = Gauss1DFitter()

R_params = dict(gfit(Rhist, x=Rcenters, sigmax=100))
R_best_fit = gfit.fitfunc(Rcenters, **R_params)

G_params = dict(gfit(Ghist, x=Gcenters, sigmax=100))
G_best_fit = gfit.fitfunc(Gcenters, **G_params)

B_params = dict(gfit(Bhist, x=Bcenters, sigmax=100))
B_best_fit = gfit.fitfunc(Bcenters, **B_params)

Grey_params = dict(gfit(Greyhist, x=Greycenters, sigmax=100))
Grey_best_fit = gfit.fitfunc(Greycenters, **Grey_params)

Nsigmas = 10
figure(2);clf();
subplot(2,2,1)
plot(Rcenters, Rhist)
plot(Rcenters, R_best_fit)
Rmin = R_params['xc']-R_params['sigmax']*5
Rmax= R_params['xc']+R_params['sigmax']*5
xlim(Rmin, Rmax)

subplot(2,2,2)
plot(Gcenters, Ghist)
plot(Gcenters, G_best_fit)
Gmin = G_params['xc']-G_params['sigmax']*5
Gmax= G_params['xc']+G_params['sigmax']*5
xlim(Gmin, Gmax)

subplot(2,2,3)
plot(Bcenters, Bhist)
plot(Bcenters, B_best_fit)
Bmin = B_params['xc']-B_params['sigmax']*5
Bmax= B_params['xc']+B_params['sigmax']*5
xlim(Bmin, Bmax)

subplot(2,2,4)
plot(Greycenters, Greyhist)
plot(Greycenters, Grey_best_fit)
Greymin = Grey_params['xc']-Grey_params['sigmax']*5
Greymax= Grey_params['xc']+Grey_params['sigmax']*5
xlim(Greymin, Greymax)

# now normalize each image
def normalize(img, mn, mx, logimg=True):
    ''' normalize to a uint8 
        This is also known as byte scaling.
    '''
    dynamic_range = 2**8-1
    if logimg:
        img = np.log10(img)
        mn = np.log10(mn)
        mx = np.log10(mx)
    img = img-mn
    img /= (mx-mn)
    img *= dynamic_range
    img = np.minimum(np.maximum(img, 0), dynamic_range)
    img = img.astype(np.uint8)
    return img


Rnorm = normalize(R, Rmin, Rmax)
Gnorm = normalize(G, Gmin, Gmax)
Bnorm = normalize(B, Bmin, Bmax)
Greynorm = normalize(Grey, Greymin, Greymax)

set_cmap("Greys_r")
figure(4);clf();
imshow(LGrey[200:900,600:1300],vmin=3.8,vmax=3.9)

figure(5);clf();
imshow(Grey[200:900,600:1300],vmin=10**3.8,vmax=10**3.9)

img_grey = normalize(Grey[200:900, 600:1300], 10**3.8, 10**3.9)
limg_grey = normalize(LGrey[200:900, 600:1300], 3.8, 3.9)

from PIL import Image
img = Image.fromarray(img_grey)
limg = Image.fromarray(limg_grey)

# saving 

#class JPEGImages:
#    def __init__(self, fpath, imgnums, prefix="IMG_", suffix=".JPG"):
#        self.prefix = prefix
#        self.suffix = suffix
#        self.imgnums = imgnums
#        self.fpath = fpath
#        self.number_files = len(imgnums)
#
#    def __getitem__(self, n):
#        #if n < self.first_file or n > self.first_file + self.number_files:
#        if n < 0 or n > self.number_files:
#            raise KeyError
#        filename = "{}/{}{}{}".format(self.fpath,self.prefix, self.imgnums[n],self.suffix)
#        return np.array(Image.open(filename))
#
#    def __len__(self):
#        return self.number_files
#
#
#from skbeam.core.utils import radial_grid, angle_grid
#
## x0, x1, y0, y1
#def reduce_array(img):
#    ''' bounds global for now'''
#    bounds = [-20, 20, -20, 20]
#    shape = img.shape
#    x0, x1 = shape[1]//2+ bounds[0], shape[1]//2 + bounds[1]
#    y0, y1 = shape[0]//2+ bounds[2], shape[0]//2 + bounds[3]
#    img = img[y0:y1, x0:x1]
#    return img
#
#imgs = JPEGImages(PDIR, whirlpool_nums)
#Nimgs = len(imgs)
#
## list of lists of dictionaries with best params
#best_fits = list()
#'''
#
#    1. Instantiate CrossCorrelator, mask
#    2. loop over imgs
#        1. cross correlate img[a], img[b]
#        2. fit cross correlation
#        3. save best_fit values
#
#
#'''
#
## get first image
#img = imgs[0][:,:,0]
#from skbeam.core.correlation import CrossCorrelator
##ccorr_nosym = CrossCorrelator(img.shape, mask=bind*conf.mask, normalization='regular')
##cctot_nosym = ccorr_nosym(imgs[0], imgs[200])
##cc1_nosym = cctot_nosym[10]
#
## xl, xr, ylow, yupper
#imgbounds = [1600, 2910, 786, 1630]
#img = imgs[0][imgbounds[0]:imgbounds[1], imgbounds[2]:imgbounds[3],0]
#ccorr_sym = CrossCorrelator(img.shape,mask=np.ones_like(img), normalization='symavg')
##### 2. loop over images
## maybe outer product of lists instead?
#from tools.Fitters2D import Gauss2DFitter
#
#frame0 = 0 #frame ot compare to
#Nframes = Nimgs-frame0
#gfit = Gauss2DFitter()
#img1 = imgs[0][imgbounds[0]:imgbounds[1], imgbounds[2]:imgbounds[3], 0]
#for i in range(1, Nframes):
#    fitlist = list()
#    img2 = imgs[i][imgbounds[0]:imgbounds[1], imgbounds[2]:imgbounds[3], 0]
#    cctot_sym = ccorr_sym(img1, img2)
#    Nres = len(cctot_sym)
#    cc1_sym = reduce_array(cctot_sym)
#    best_values_sym = gfit(cc1_sym)
#    best_fits.append(best_values_sym)
#
#def extract_param(bestfits, key):
#    Nframes = len(bestfits)
#    params = np.zeros((Nframes))
#    for i in range(Nframes):
#        params[i] = bestfits[i][key]
#    return params
#
#xcs = np.array(extract_param(best_fits, 'xc'))
#xcs = [0].append(xcs)
#ycs = np.array(extract_param(best_fits, 'yc'))
#ycs = [0].append(ycs)
#
#
#def registack(imgs, xshifts, yshifts):
#    ''' register and stack images.'''
#    imgstacked = np.zeros_like(imgs[0], dtype=float)
#    for i in len(imgs):
#        xshift = int(xshifts[i])
#        yshift = int(yshifts[i])
#        imgstacked += np.roll(np.roll(imgs[i],xshift,axis=1),yshift, x=0)
#    return imgstacked
#
