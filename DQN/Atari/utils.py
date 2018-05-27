import numpy as np
from skimage.transform import resize
from collections import deque

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
      img_resized (2D array) : Rescaled luminance image uint8 np array .
    '''

    # Convert RGB -> Luminance
    R = 0.2126
    G = 0.7152
    B = 0.0722
    img_lum = R * img_rgb[:,:,0] + G * img_rgb[:,:,1] + B * img_rgb[:,:,2]

    #return img_lum
    # Resize image
    img_resized = resize(img_lum, (img_resize_w_px, img_resize_h_px), mode="constant")

    return img_resized.astype("uint8")


class FrameBuffer():
    '''Frame buffer object for stacking frames when doing inference (i.e. predicting Q-value for 'a').

    Uses a 'deque' buffer object, meaning that appending new elements in the back pushes out old elements
    from the front, keeping the buffer size constant.

    Class methods:
      store  : Add new frame to the back of the buffer, pushing out the frame in front.
      sample : Return all stored frames as 3D np array.
      reset  : Reset the buffer with new frames at the start of an episode.
    '''
    def __init__(self, size = 4):

        self.buffer = deque(maxlen=size)
        self.buffer_size = size

    def store(self, frame):
        # Add frame to the back, pushing out the frame in the front
        # [1,2,3,4].append(0) -> [2,3,4,0]
        self.buffer.append(frame)

    def sample(self):
        # Return all buffered frames as a 3D array
        frame_1 = self.buffer[0]
        frame_2 = self.buffer[1]
        frame_3 = self.buffer[2]
        frame_4 = self.buffer[3]
        frame_stack = np.stack((frame_1, frame_2, frame_3, frame_4), axis=2)
        return frame_stack

    def reset(self, env):
        for i in range(self.buffer_size):
            s = env.reset()
            s = preprocessImage(s)
            self.buffer.append(s)
