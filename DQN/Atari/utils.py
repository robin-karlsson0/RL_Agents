import numpy as np
from skimage.transform import resize


def preprocessImage(img_rgb, img_resize_w_px=84, img_resize_h_px=84):
    '''Process the raw Atari input image the same way as was done in the DQN Nature paper.
    
    Process:
    1. Convert the three RGB channels into a Luminance channel.
    2. Shrink the image to 84x84 pixels.

    Args:
      img_rgb (3-layer array) : RGB input image.
      img_resize_w_px (int)   : Pixel width of rescaled image.
      img_resize_h_px (int)   : Pixel height of rescaled image.

    Returns:
      img_resized (2D array) : Rescaled luminance image np array.
    '''

    # Convert RGB -> Luminance
    R = 0.2126
    G = 0.7152
    B = 0.0722
    img_lum = R * img_rgb[:,:,0] + G * img_rgb[:,:,1] + B * img_rgb[:,:,2]

    #return img_lum
    # Resize image
    img_resized = resize(img_lum, (img_resize_w_px, img_resize_h_px), mode="constant")
    return img_resized


