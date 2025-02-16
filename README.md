# Multiespectral correction

This set of scripts is intended to check and correct mialignments and distortion issues when working with multiespectral images that have been taken with different cameras and are not calibrated. 


## Scripts available:

- [ ] `00_review_image_corresp_handeye.py`: This tool lets you match manually corresponding pairs of points between images (one LWIR and its corresponding RGB image).
- [ ] `01_computeImageMovement.py`: Making use of optical flow computes the transform between contiguous RGB images. 
- [ ] `02_computeMultiespectralDistortionFactor.py`: The transform between LWIR and RGB images should be some function of the transform between contiguous RGB images. This computes that factor along with other useful data.
- [ ] `03_computeTransform.py`: Computes transform for each image in the dataset. 
- [ ] `04_projectImages.py`: Small script to debug the correction projecting images over its corrected/non-corrected pairs.

## Toolchain

In some multiespectral datasets it might happen that images do not match in terms of distortion. As usually we don't have access to the camera nor to the calibration data (which would be the ideal). As we have two images some calibration might be corrected based on matching points between both of them.

In this case we will compute the relative distortion of LWIR image with regards to RGB image.
<p align="center">
    <figure style="display: inline-block; text-align: center; margin: 0 10px;">
        <img src="images/lwir_I01689.png" alt="LWIR imag used as reference" width="80%">
        <figcaption>LWIR Image Used as Reference</figcaption>
    </figure>
    <figure style="display: inline-block; text-align: center; margin: 0 10px;">
        <img src="images/visible_I01689.png" alt="RGB imag used as reference" width="80%">
        <figcaption>RGB Image Used as Reference</figcaption>
    </figure>
</p>

With any of the `review_image_corresp_*.py` scripts we will match corresponding points between both images. The script goes iteratively over the dataset picking random images (without repetition), so its not very important to have a lot of points in a single image-pair, only those that are a sure match. The raw data is stored in a yaml as in `data/he_points.yaml` with all the matching points between images.

Once we have enough points we can apply some filter to them (we have to tackle possible human error in the tagging) an interpolate a grid of distortion vectors between images (a distortion vector has a starting point with the coordinates in LWIR image and an arrow reaching the matching point coordinates in RGB image):

<p align="center">
    <figure style="display: inline-block; text-align: center; margin: 0 20px;">
        <img src="plot/calib_arrows_wremoved.png" alt="Distortion vectors. In blue the ones that will be used; in red the filtered ones" width="80%">
        <figcaption>Raw distortion vectors from tagged data. In red vectors that are filtered and won't be used.</figcaption>
    </figure>
    <figure style="display: inline-block; text-align: center; margin: 0 20px;">
        <img src="plot/calib_arrows_averagefilter.png" alt="Interpolated grid of distortion vectors to be used" width="80%">
        <figcaption>Interpolated grid of distortion vectors to be used</figcaption>
    </figure>
</p>

With this filtered matching point grid we can perform a calibration of the relative distortion between both images and hope to correct it as much as possible.
