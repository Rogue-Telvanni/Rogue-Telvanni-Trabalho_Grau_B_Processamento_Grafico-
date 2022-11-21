import cv2 as cv
import numpy as np


class Image_Filtering:
    def __init__(self):
        self.filter_type = 0
        self.foreground_image = 0
        self.stickers_position = []

    def set_filter_type(self, filter_type):
        self.filter_type = filter_type

    def set_foreground_type(self, foreground_image):
        self.foreground_image = foreground_image

    def reset(self):
        self.filter_type = 0
        self.foreground_image = 0
        self.stickers_position = []

    def add_filter(self, image, is_negative, has_color_filter, color_filter_channels):
        # why there are now switch case
        if self.filter_type == 1:
            image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            if is_negative:
                image_grayscale = cv.bitwise_not(image_grayscale)
            return image_grayscale
        elif self.filter_type == 2:
            image_blurred15x15 = cv.blur(image, (15, 15))

            if has_color_filter and color_filter_channels is not None:
                im_blue, im_green, im_red = cv.split(image_blurred15x15)
                image_blurred15x15 = cv.merge([im_blue | color_filter_channels[0][0],
                                               im_green | color_filter_channels[0][1],
                                               im_red | color_filter_channels[0][2]])

            if is_negative:
                image_blurred15x15 = cv.bitwise_not(image_blurred15x15)
            return image_blurred15x15
        elif self.filter_type == 3:
            image_gaussian_blurred15x15 = cv.GaussianBlur(image, (15, 15), 0)

            if has_color_filter and color_filter_channels is not None:
                im_blue, im_green, im_red = cv.split(image_gaussian_blurred15x15)
                image_gaussian_blurred15x15 = cv.merge([im_blue | color_filter_channels[0][0],
                                                        im_green | color_filter_channels[0][1],
                                                        im_red | color_filter_channels[0][2]])

            if is_negative:
                image_gaussian_blurred15x15 = cv.bitwise_not(image_gaussian_blurred15x15)
            return image_gaussian_blurred15x15
        elif self.filter_type == 4:
            image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image_gaussian_blurred15x15 = cv.GaussianBlur(image_grayscale, (15, 15), 0)
            imgCanny = cv.Canny(image_gaussian_blurred15x15, 50, 100)
            kernel = np.ones((3, 3), np.uint8)
            image_dilation = cv.dilate(imgCanny, kernel, iterations=1)
            return image_dilation
        elif self.filter_type == 5:
            image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image_gaussian_blurred15x15 = cv.GaussianBlur(image_grayscale, (15, 15), 0)
            imgCanny = cv.Canny(image_gaussian_blurred15x15, 50, 100)
            kernel = np.ones((3, 3), np.uint8)
            image_dilation = cv.dilate(imgCanny, kernel, iterations=1)
            image_erosion = cv.erode(image_dilation, kernel, iterations=1)
            return image_erosion
        elif self.filter_type == 6:
            _, sketch = cv.pencilSketch(image, sigma_s=40, sigma_r=0.6, shade_factor=0.01)

            if has_color_filter and color_filter_channels is not None:
                im_blue, im_green, im_red = cv.split(sketch)
                sketch = cv.merge([im_blue | color_filter_channels[0][0], im_green | color_filter_channels[0][1],
                                   im_red | color_filter_channels[0][2]])

            if is_negative:
                sketch = cv.bitwise_not(sketch)

            return sketch
        elif self.filter_type == 7:
            stylized = cv.stylization(image, sigma_s=15, sigma_r=0.55)
            if has_color_filter and color_filter_channels is not None:
                im_blue, im_green, im_red = cv.split(stylized)
                stylized = cv.merge([im_blue | color_filter_channels[0][0], im_green | color_filter_channels[0][1],
                                     im_red | color_filter_channels[0][2]])
            if is_negative:
                stylized = cv.bitwise_not(stylized)
            return stylized
        else:
            if has_color_filter and color_filter_channels is not None:
                im_blue, im_green, im_red = cv.split(image)
                image = cv.merge([im_blue | color_filter_channels[0][0], im_green | color_filter_channels[0][1],
                                  im_red | color_filter_channels[0][2]])
            if is_negative:
                image = cv.bitwise_not(image)
            return image

    def add_image_overlays(self, background):
        for sticker in self.stickers_position:
            overlay, fx, fy = self.get_image_path(sticker[0])
            background = self.add_image_overlay(background, overlay, fx, fy, sticker[1], sticker[2])

        return background

    @staticmethod
    def get_image_path(sticker_type):
        if sticker_type == 1:
            return "eyeglasses.png", 0.65, 0.65
        elif sticker_type == 2:
            return "neon_hand.png", 0.2, 0.2
        elif sticker_type == 3:
            return "et_neon.png", 0.2, 0.2
        elif sticker_type == 4:
            return "chacol.png", 0.65, 0.65

    @staticmethod
    def add_image_overlay(background, overlay, fx, fy, x_offset=None, y_offset=None):
        if overlay is None:
            return background
        else:
            foreground = cv.imread(overlay, cv.IMREAD_UNCHANGED)
            foreground = cv.resize(foreground, (0, 0), fx=fx, fy=fy)

        bg_h, bg_w, bg_channels = background.shape
        fg_h, fg_w, fg_channels = foreground.shape

        assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
        assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

        # center by default
        if x_offset is None:
            x_offset = (bg_w - fg_w) // 2
        if y_offset is None:
            y_offset = (bg_h - fg_h) // 2

        w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
        h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

        if w < 1 or h < 1:
            return

        # clip foreground and background images to the overlapping regions
        bg_x = max(0, x_offset)
        bg_y = max(0, y_offset)
        fg_x = max(0, x_offset * -1)
        fg_y = max(0, y_offset * -1)
        foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
        background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

        # separate alpha and color channels from the foreground image
        foreground_colors = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

        # construct an alpha_mask that matches the image shape
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

        # combine the background with the overlay image weighted by alpha
        composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

        # overwrite the section of the background image that has been updated
        background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
        out = background.copy()
        return out
