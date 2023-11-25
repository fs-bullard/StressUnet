<<<<<<< HEAD
# StressUnet
predict stress field in photoelasticity

Once downloaded:
- run the following command in the command prompt:
```py train_val.py```
- This should train the model. Once complete, there should be a new folder called unet with lots files in
- 

Note: I had to change `window_size/2` to `window_size//2` in the library:

```
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
```
=======
# StressUnet - extracting stress maps from isochromatic images with deep learning
 

To train the model (Windows):

- Run the following in command prompt:
    ```
    venv\Scripts\activate.bat
    pip install -r requirements.txt
    ```

 - Replace `venv\Lib\pytorch_ssim\__init__.py` with `replacement-files\__init__.py` to fix the integer division bug

 - Run `train_val.py` 


>>>>>>> bf99fa5afaa16149f2bfe3b143164f4339c3e98f
