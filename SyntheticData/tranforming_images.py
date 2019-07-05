%pylab

import pytiff
Fl = []
with pytiff.Tiff("cell1_1.tif") as handle:
    Flt = zeros((128,1024))
    for i,page in enumerate(handle):
        Flt += page 
Fl = array(Fl)

miFt = min(Flt.flatten())
maFt = max(Flt.flatten())
Ft = array(Flt[:,:512],dtype=float)
imshow((Ft-miFt)/(maFt-miFt))

fftF = fftshift(fft2(Ft))

miff = min(log(abs(fftF)).flatten())
maff = max(log(abs(fftF)).flatten())
imshow((log(abs(fftF))-miff)/(maff-miff))

hist(log(abs(fftF)).flatten(),bins=41)

imshow((Ft-miFt)/(maFt-miFt))
figure(2)
fftF2 = fftF*(1+randn(fftF.shape[0],fftF.shape[1])*0.15); Ft2 = real(ifft2(ifftshift(fftF2))); miFt2 = min(Ft2.flatten()); maFt2 = max(Ft2.flatten()); imshow((Ft2-miFt2)/(maFt2-miFt2))
fftF2 = fftF*(1+randn(fftF.shape[0],fftF.shape[1])*0.85); Ft2 = real(ifft2(ifftshift(fftF2))); miFt2 = min(Ft2.flatten()); maFt2 = max(Ft2.flatten()); imshow((Ft2-miFt2)/(maFt2-miFt2))
