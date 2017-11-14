"""
=================================================
Non-local means denoising for preserving textures
=================================================

In this example, we denoise a detail of the astronaut image using the non-local
means filter. The non-local means algorithm replaces the value of a pixel by an
average of a selection of other pixels values: small patches centered on the
other pixels are compared to the patch centered on the pixel of interest, and
the average is performed only for pixels that have patches close to the current
patch. As a result, this algorithm can restore well textures, that would be
blurred by other denoising algoritm.
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma


astro = img_as_float(data.astronaut())
astro = astro[30:180, 150:300]

sigma = 0.15
noisy = astro + sigma * np.random.standard_normal(astro.shape)
noisy = np.clip(noisy, 0, 1)

sigma_est = np.mean(estimate_sigma(noisy, multichannel=True))
print("sigma_est = {}".format(sigma_est))

denoise = denoise_nl_means(noisy, 7, 9, h=0.66*sigma_est, multichannel=True,
                           sigma=sigma_est, fast_mode=True)

fig, ax = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

ax[0].imshow(noisy)
ax[0].axis('off')
ax[0].set_title('noisy')
ax[1].imshow(denoise)
ax[1].axis('off')
ax[1].set_title('non-local means')

fig.tight_layout()

plt.show()
