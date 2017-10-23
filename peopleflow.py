#     Peopleflow, a computer vision program that computes the flow of people
#     Copyright (C) 2017  Gianluca Guidi
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


import ast
import configparser
import csv
import os
import sys
from collections import deque

import cv2
import numpy as np

__author__ = 'Gianluca Guidi'


class Config(object):
    """
    Configuration object. Allows reading configuration from a .ini file.
    """

    video_path_key = 'video_path'
    region_map_path_key = 'region_map_path'
    background_path_key = 'background_path'
    pix_people_path_key = 'pix_people_path'
    tot_count_gtruth_path_key = 'tot_count_gtruth_path'
    reset_path_key = 'reset_path'
    box_key = 'box'
    init_directed_count_key = 'init_directed_count'
    p_key = 'p'
    frame_interval_key = 'frame_interval'
    averaging_interval_key = 'averaging_interval'
    histogram_matching_key = 'histogram_matching'
    blur_function_key = 'blur_function'
    blur_kernel_size_key = 'blur_kernel_size'
    blur_sigma_key = 'blur_sigma'
    start_at_key = 'start_at'
    stop_at_key = 'stop_at'
    show_contours_key = 'show_contours'
    results_path_key = 'results_path'
    play_delay_key = 'play_delay'
    rewind_delay_key = 'rewind_delay'
    pause_frame_interval_key = 'pause_frame_period'
    reset_period_key = 'reset_period'

    def __init__(self, filepath):
        """
        Read configuration from file and build the config object accordingly.
        :param filepath: path to config file
        """

        self.config = configparser.ConfigParser()
        self.config.read(filepath)
        top = self.config[self.config.sections()[0]]

        if Config.video_path_key in top:
            self.video_path = top[Config.video_path_key]
        else:
            print('Configuration requires video path')
            raise FileNotFoundError

        self.region_map_path = top.get(Config.region_map_path_key, None)
        self.background_path = top.get(Config.background_path_key, None)
        self.pix_people_path = top.get(Config.pix_people_path_key, None)
        self.tot_count_gtruth_path = top.get(Config.tot_count_gtruth_path_key, None)
        self.results_path = top.get(Config.results_path_key, None)
        self.reset_path = top.get(Config.reset_path_key, None)

        box = top.get(Config.box_key, None)
        if box is not None:
            self.box = ast.literal_eval(top[Config.box_key])

        self.init_directed_count = top.get(Config.init_directed_count_key, None)
        if self.init_directed_count is not None:
            self.init_directed_count = np.array(ast.literal_eval(self.init_directed_count))

        self.p = ast.literal_eval(top.get(Config.p_key, '[0.5, 0.5]'))
        assert (0 < self.p[0] <= 1) and (0 < self.p[1] <= 1)

        self.frame_interval = top.getint(Config.frame_interval_key, 90)
        assert self.frame_interval > 0

        self.averaging_interval = top.getint(Config.averaging_interval_key, 5)
        assert self.averaging_interval < self.frame_interval and self.averaging_interval % 2 == 1

        self.histogram_matching = top.getboolean(Config.histogram_matching_key, True)

        # Read blur function and parameters from config file
        blur_function = top.get(Config.blur_function_key, '')
        self.blur_kernel_size = ast.literal_eval(top.get(Config.blur_kernel_size_key, 'None'))
        self.blur_sigma = ast.literal_eval(top.get(Config.blur_sigma_key, 'None'))

        # Define different blur wrapper functions.
        def average_blur(img):
            return cv2.blur(img, self.blur_kernel_size)

        def gaussian_blur(img):
            return cv2.GaussianBlur(img, self.blur_kernel_size,
                                    self.blur_sigma[0], self.blur_sigma[1])

        def median_blur(img):
            return cv2.medianBlur(img, self.blur_kernel_size)

        def bilateral_blur(img):
            return cv2.bilateralFilter(img, self.blur_kernel_size, self.blur_sigma)

        if blur_function == 'average':
            self.blur = average_blur
        elif blur_function == 'gaussian':
            self.blur = gaussian_blur
        elif blur_function == 'median':
            self.blur = median_blur
        elif blur_function.strip() == 'bilateralfilter':
            self.blur = bilateral_blur
        else:
            self.blur = lambda x: x

        self.start_at = top.getint(Config.start_at_key, 0)
        # this stop at value allows for 207 days of video at 120 fps on a 32 bit system
        maxnum = sys.maxsize
        if maxnum < 2**31-1:
            maxnum = 2**31-1
        self.stop_at = top.getint(Config.stop_at_key, maxnum)

        self.show_contours = top.getboolean(Config.show_contours_key, False)
        self.play_delay = top.getint(Config.play_delay_key, 1)
        self.rewind_delay = top.getint(Config.rewind_delay_key, 30)
        self.pause_frame_interval = top.getint(Config.pause_frame_interval_key, 4500)
        self.reset_period = top.getint(Config.reset_period_key, 1)


class RegionMap(object):
    """
    A RegionMap object can be used to identify the regions withing an image.
    Regions, accessible through the regions field, are used to compute the flow of people.
    Several useful fields are generated upon initialization:
    regions: a list of boolean masks, one for each region;
    foreground: a sum of all regions;
    background: the opposite of foreground, that is the uninteresting part of the image;
    region_countours: region contour list
    bounding_box: bounding box of all the regions joined together
    """

    def __init__(self, regmap):
        """
        Create a RegionMap object based on the given image.
        The input is a grayscale image with background pixels set to black.
        Regions will be indexed based on the pixel intensities that define them,
        e.g. the region with the lowest non-zero intensity will be the first.
        """
        super().__init__()
        self.regmap = regmap
        hist, bins = np.histogram(self.regmap.ravel(), 256, [0, 256])
        self.background = self.regmap == 0
        self.foreground = self.regmap != 0
        self.regions = []
        for grey_level in range(1, len(hist)):
            count = hist[grey_level]
            if count > 0:
                self.regions.append(self.regmap == grey_level)
        # Detect and store region contours.
        # OpenCV wants white contours over black background, so...
        self.region_contours = []
        for region in self.regions:
            binary_region = np.zeros(self.regmap.shape, np.uint8)  # Black image
            binary_region[region] = 255  # Paint region white
            contour = cv2.findContours(binary_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1][0]
            self.region_contours.append(contour)
        # Additionally compute and store the bounding box of all the regions joined together
        binary_regions = np.zeros_like(self.regmap)
        binary_regions[self.foreground] = 255
        contour = cv2.findContours(binary_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1][0]
        min_x, min_y, w, h = cv2.boundingRect(contour)
        self.bounding_box = ((min_x, min_y), (min_x+w, min_y+h))


class FittingPoly(object):
    """
    Fit pixel and people count data with a polynomial by using least squares.
    
    n different polynomials will be produced, one for each region.
    The resulting function has to be called with an index referring to
    the intended region.
    """

    def __init__(self, filename, degree=1):
        """
        :param filename: the path to a csv file in the following format:
        
            framenum,,pixcount_1:peoplecount_1,,pixcount_n:peoplecount_n
            
            where framenum is the frame number, pixcount_x is the number of 
            foreground pixels for region x, and peoplecount is the number of
            people for region x.
        :param degree: the degree of the polynomial.
        """
        f = open(filename, 'r')
        line = f.readline()
        relations = line.strip().split(',,')
        regions = len(relations)-1  # excluding frame number
        # 0 pixels always means 0 people
        pixcounts = [[0.] for i in range(regions)]
        peoplecounts = [[0.] for i in range(regions)]
        while True:
            line = f.readline()
            if not line or line == '\n':
                break
            relations = line.strip().split(',,')
            del relations[0]  # discard frame number
            # check that entry is valid:
            valid = True
            for relation in relations:
                if relation[-1] == ':':  # people count absent
                    valid = False
                    break

            if valid:
                for i in range(regions):
                    pix, people = relations[i].split(':')
                    pixcounts[i].append(float(pix))
                    peoplecounts[i].append(float(people))
        f.close()
        self.p = []
        for i in range(regions):
            peoplecounts[i] = np.array(peoplecounts[i])
            pixcounts[i] = np.array(pixcounts[i])
            polynomial_coeffs = np.polyfit(pixcounts[i], peoplecounts[i], degree)
            polynomial_coeffs = np.flip(polynomial_coeffs, 0)  # increasing degree order
            self.p.append(polynomial_coeffs)
        print('Created polynomial with degree %d based on %d entries.' % (degree, len(peoplecounts[0])))

    def __call__(self, *args, **kwargs):
        """
        Compute number of people based on foreground pixels x at region k.
        :param args: The first arg is x, the number of pixels. The second arg is k, the region index.
        :return: Number of people present in the k-th region.
        """
        x = args[0]
        k = args[1]
        result = 0
        pow_x = 1
        for i in range(0, len(self.p[k])):
            coeff = self.p[k][i]
            result += coeff * pow_x
            pow_x *= x
        if result < 0:
            result = 0
        return result


class NoFrameException(Exception):
    """
    Raised when a frame could not be read from a video capture.
    Usually raised at end of said capture.
    """
    pass


class ImageStream(object):
    """
    Class to easily read frames in png format.
    Images must be in the same directory and are selected based on frame number.
    """

    def __init__(self, dirpath, fileprefix):
        """
        Initialize an ImageStream instance by specifying the common
        dirpath and fileprefix of images to read or write.
        :param dirpath: path of directory in which frame images are (or will be) stored.
        :param fileprefix: prefix of all frame image file names; it is always followed
         by a 6 digits 0-padded frame number.
        """
        self.dirpath = dirpath
        self.fileprefix = fileprefix

    def get_path(self, frame_num):
        filename = '%s%06d.png' % (self.fileprefix, frame_num)
        return os.path.join(self.dirpath, filename)

    def read(self, frame_num):
        return cv2.imread(self.get_path(frame_num))

    def write(self, image, frame_num):
        cv2.imwrite(self.get_path(frame_num), image)


class BackgroundSubtractorAdaptiveMedian(object):
    """
    A background subtractor based on the adaptive median algorithm.
    """

    def __init__(self, sampling_rate, learning_frames, model, roi=None, region_map=None):
        """
        Initialize a background subtractor object. The apply() method should be called on each frame.
        :param sampling_rate: the background model is updated every sampling_rate frames.
        :param learning_frames: the model is updated at every frame for this number of frames, regardless of the sampling rate.
        :param model: a grayscale image to initialize the background model.
        :param roi: a boolean mask identifying the ROI, used when creating the foreground mask.
        """
        self.sampling_rate = sampling_rate
        self.learning_frames = learning_frames
        self.frames_seen = 0
        self.bg_model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            self.roi = roi
        else:
            self.roi = np.ones(model.shape, np.bool)
        self.region_map = region_map

    def apply(self, frame):
        self.frames_seen += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        absdiff = cv2.absdiff(frame, self.bg_model)

        # Use Otsu on the ROI instead of using a fixed threshold.
        # This can alleviate some problems caused by an inexact background model.
        fg_mask = np.zeros_like(absdiff)
        # for roi in self.region_map.regions:
        #     self.roi = roi
        ret, diff = cv2.threshold(absdiff[self.roi], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        diff = diff.reshape((diff.shape[0],))  # for some reason OpenCV translates the matrix, so we correct it
        fg_mask[self.roi] = diff

        # Update model at the given sampling rate.
        if self.frames_seen % self.sampling_rate == 0 or self.frames_seen <= self.learning_frames:
            to_increase = self.bg_model < frame
            self.bg_model[to_increase] += 1
            to_decrease = self.bg_model > frame
            self.bg_model[to_decrease] -= 1
        return fg_mask

    def get_background_image(self):
        return self.bg_model


class BackgroundSubtractorMOG2(object):
    """
    Simple wrapper of OpenCV's MOG2 background subtractor; just for convenience.
    """

    def __init__(self, model=None, roi=None):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.bg_subtractor.setShadowValue(0)
        self.bg_subtractor.apply(model)
        if roi is not None:
            self.roi = roi
        else:
            self.roi = np.ones(model.shape, np.bool)

    def apply(self, frame):
        return self.bg_subtractor.apply(frame)

    def get_background_image(self):
        return self.bg_subtractor.getBackgroundImage()


class PostProcessor(object):
    """
    Object used to apply some morphological operations to a foreground mask.
    These operations are useful for a specific scenario and will probably have to be adapted for a different one.
    """

    def __init__(self, region_map):
        self.vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        self.point = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        self.cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        self.region_map = region_map

    def process(self, fg_mask):
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_ERODE, self.vertical)
        # Detect blobs.
        fg_mask, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Erase small blobs by painting them black.
        for contour in contours:
            if cv2.contourArea(contour) <= 2:  # blobs of 2 pixels or less are likely just noise
                fg_mask = cv2.drawContours(fg_mask, [contour], -1, 0, cv2.FILLED)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, self.vertical)

        if len(self.region_map.regions) == 3:
            reg2 = self.region_map.regions[2]
            fg_mask2 = np.copy(fg_mask)
            fg_mask2 = cv2.morphologyEx(fg_mask2, cv2.MORPH_CLOSE, self.cross)
            fg_mask[reg2] = fg_mask2[reg2]

        return fg_mask


class ResultWriter(object):
    """
    Writes results in the output file.
    """
    def __init__(self, filename, with_gtruth=False):
        """
        Initialize ResultWriter instance with the output file name.
        :param filename: path to results file.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file, quoting=csv.QUOTE_NONNUMERIC)
        if with_gtruth:
            self.writer.writerow(('frame', 'frame_from_left_gtruth', 'frame_from_right_gtruth', 'from_left_gtruth',
                                  'from_left', 'from_right_gtruth', 'from_right'))
        else:
            self.writer.writerow(('frame', 'from_left', 'from_right'))

    def write(self, *args):
        """
        Write arguments as a row in the results file.
        :param args: arguments to be written
        """
        self.writer.writerow(args)

    def close(self):
        """
        Close the file after results have been written.
        """
        self.file.close()


def read_frame(cap):
    """
    Read frame from a cv2.VideoCapture instance. Raises NoFrameException if
    no frame could be read.

    :param cap: an object created with cv2.VideoCapture
    :return: the frame that was read
    """
    ok, frame = cap.read()
    if not ok:
        raise NoFrameException
    return frame


def get_region_colors(region_map):
    """
    Select distant hues at maximum saturation for each region.

    :param region_map: RegionMap instance
    :return: A list of colors, one for each region.
    """
    if len(region_map.regions) == 1:
        return [(0, 0, 255)]
    else:
        region_colors = []
        step = 255 / (len(region_map.regions) - 1)
        for i in range(0, len(region_map.regions)):
            hsv_color = np.uint8([[[step * i, 255, 255]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
            region_colors.append(bgr_color.tolist()[0][0])
        return region_colors


def draw_region_borders(img, region_map, colors=[]):
    """
    Draws the region borders on the input image, with optional specified colors.
    :param img: image to draw borders to
    :param region_map: RegionMap instance that defines the regions
    :param colors: list of colors to draw the borders with, one for each region
    :return: image with region borders drawn
    """
    if len(colors) == 0:
        for region in region_map.regions:
            colors.append((0, 0, 255))
    for i in range(len(region_map.regions)):
        frame = cv2.drawContours(img, region_map.region_contours, i, colors[i], 1)


def generate_ground_truth(config):
    """
    Generate frame images and template file to hold ground truth values.
    :param config: Configuration object.
    """

    min_point, max_point = config.box
    min_x, min_y = min_point
    max_x, max_y = max_point

    cap = cv2.VideoCapture(config.video_path)

    # Open region file.
    regmap = cv2.imread(config.region_map_path, cv2.IMREAD_GRAYSCALE)
    regmap = regmap[min_y:max_y, min_x:max_x]
    region_map = RegionMap(regmap)
    # Create background subtractor object.
    background = cv2.imread(config.background_path)
    background = background[min_y:max_y, min_x:max_x]
    bgs = BackgroundSubtractorAdaptiveMedian(12, 45, background, region_map.foreground)
    post_processor = PostProcessor(region_map)

    # Region border colors to be drawn at each frame.
    region_colors = get_region_colors(region_map)

    large_regmap = cv2.resize(regmap, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    large_region_map = RegionMap(large_regmap)

    frame_num = -1
    try:
        dirpath = os.path.dirname(config.pix_people_path)
        os.makedirs(dirpath, exist_ok=True)
        with open(config.pix_people_path, 'w') as f:
            # Write header
            f.write('frame_num')
            for i in range(len(region_map.regions)):
                f.write(',,pixcount_%d:peoplecount_%d' % (i, i))
            f.write('\n')

            # Skip to start at desired frame
            frame_num += skip_frames(cap, config.start_at, bgs, config.box)

            maxframe = config.stop_at
            while frame_num <= maxframe:
                frame = read_frame(cap)
                frame_num += 1

                # Crop and pre-process
                frame = frame[min_y:max_y, min_x:max_x]
                fg_mask = bgs.apply(frame)

                if frame_num % config.frame_interval == 0:
                    fg_mask = post_processor.process(fg_mask)
                    # Estimate people count in section based on number of foreground pixels.
                    pix_count = []
                    for section in region_map.regions:
                        pix_count.append(np.count_nonzero(fg_mask[section]))

                    # Generate frame image and write to area : people count file.
                    fg_mask = cv2.resize(fg_mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                    fg_mask, large_contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    people_contours = []
                    large_people_contours = [x for x in large_contours]
                    f.write(str(frame_num))
                    for count in pix_count:
                        f.write(',,' + str(count) + ':')
                    f.write('\n')

                    # Enlarge frame to make it more visible
                    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

                    draw_region_borders(frame, large_region_map, region_colors)

                    # Draw a copy which includes people countours (in red)
                    frame = cv2.drawContours(frame, large_people_contours, -1, (0, 0, 255), 1)
                    # Redraw region borders to overwrite people contours which are crossing regions
                    draw_region_borders(frame, large_region_map, region_colors)

                    # Draw frame.
                    cv2.imshow('frame', frame)
                    k = cv2.waitKey(1)
                    if k == 27:  # Esc
                        break
    except NoFrameException:
        pass
    finally:
        cap.release()
    cv2.destroyAllWindows()
    print('Stopped at frame ', frame_num)


class ResetTable(object):
    """
    Provides access to resetting data.
    """
    def __init__(self, filepath, region_map, period, starting_frame):
        self.table = []
        with open(filepath, 'r') as f:
            f.readline()  # discard header
            line = f.readline()
            while line != '':
                frame_num, directed_count = line.split(';')
                frame_num = int(frame_num.strip())
                directed_count = ast.literal_eval(directed_count.strip())
                # Remove additional counts for unused regions
                directed_count = directed_count[:len(region_map.regions)]
                self.table.append((frame_num, directed_count))
                line = f.readline()
        self.table.append((-1, ()))
        self.i = 0
        self.regions = len(region_map.regions)
        self.period = period
        if self.table[self.i][0] % self.period != 0:
            self.next()
        while self.current_frame < starting_frame:
            self.next()

    def next(self):
        self.i += 1
        while self.table[self.i][0] % self.period != 0 and self.i < len(self.table) - 1:
            self.i += 1

    @property
    def current_frame(self):
        return self.table[self.i][0]

    @property
    def current_count(self):
        return np.array(self.table[self.i][1]).reshape(self.regions, 2)


def play(config):
    """
    Play video and optionally show region borders and people contours.
    :param config: Config object that must include:
    video_path: path to video file
    and optionally:
    region_map_path: path to region map image for displaying borders.
    background_path: path to background image for showing foreground contours.
    """

    min_point, max_point = config.box
    min_x, min_y = min_point
    max_x, max_y = max_point
    play_delay = config.play_delay
    pause_frame_interval = config.pause_frame_interval
    cap = cv2.VideoCapture(config.video_path)
    rewind_buffer = deque(maxlen=60)

    if config.background_path is not None:
        background = cv2.imread(config.background_path)
        background = background[min_y:max_y, min_x:max_x]
    if config.region_map_path is not None:
        regmap = cv2.imread(config.region_map_path, cv2.IMREAD_GRAYSCALE)
        regmap = regmap[min_y:max_y, min_x:max_x]
        region_map = RegionMap(regmap)
        colors = get_region_colors(region_map)

    frame_num = -1
    try:
        # Skip to start at desired point
        frame_num += skip_frames(cap, config.start_at)
        frame = read_frame(cap)
        frame = frame[min_y:max_y, min_x:max_x]
        frame_num += 1
        cv2.imshow('video', frame)
        cv2.waitKey(0)

        bgs = BackgroundSubtractorAdaptiveMedian(12, 45, background, region_map.foreground, region_map)
        post_processor = PostProcessor(region_map)

        maxframe = config.stop_at
        while frame_num <= maxframe:
            frame = read_frame(cap)
            frame_num += 1
            frame = frame[min_y:max_y, min_x:max_x]
            rewind_buffer.append(frame)

            fg_mask = bgs.apply(frame)
            fg_mask = post_processor.process(fg_mask)

            if config.show_contours:
                fg_mask, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, [c for c in contours], -1, (0, 0, 255), 1)

            if config.region_map_path is not None:
                draw_region_borders(frame, region_map, colors)

            cv2.imshow('background', bgs.get_background_image())
            cv2.imshow('video', frame)
            cv2.imshow('fg_mask', fg_mask)
            k = cv2.waitKey(play_delay)
            if frame_num % pause_frame_interval == 0:
                k = 32
            if k == 32:  # spacebar
                print('Paused at frame ', frame_num)
                k = cv2.waitKey(-1)
                while k != 32 and k != 27:
                    if k == 114:  # r key
                        rewind(rewind_buffer, config.rewind_delay)
                    k = cv2.waitKey(-1)
                print('Resumed')
            if k == 27:  # Esc
                break
    except NoFrameException:
        pass
    finally:
        cap.release()


class GroundTruth(object):

    def __init__(self, filepath, start_at=0):
        self.left_to_right, self.right_to_left = self.read_count(filepath, start_at)
        self.last_left_before_reset = start_at
        self.last_right_before_reset = start_at

    def read_count(self, filepath, start_at, period=9000):
        """
        Read ground truth from a txt file.
        :param filepath: path to file to read
        :return: (left_to_right, right_to_left) lines of the input file are grouped based
        on direction. Each line is a tuple of (frame_num, count)
        """
        left_to_right = [(start_at, 0)]
        right_to_left = [(start_at, 0)]
        current = left_to_right
        interval_counter = 0
        with open(filepath, 'r', newline='') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    line = line.strip()
                if line == 'left_to_right:':
                    current = left_to_right
                    interval_counter = 0
                elif line == 'right_to_left:':
                    current = right_to_left
                    interval_counter = 0
                elif not line:
                    continue
                else:
                    frame_num, count = line.split(',')
                    frame_num = int(frame_num)
                    count = float(count)
                    if frame_num > start_at:  # skip entries before the starting time
                        # current_interval = int((frame_num - start_at) / period)
                        # if current_interval > interval_counter:
                        #     interval_counter += 1
                        #     current.append((start_at + current_interval * period, 0))
                        #
                        # # add this count to previous one to compute prefix sum
                        # total_count = current[-1][1] + count
                        current.append((frame_num, count))

        return left_to_right, right_to_left

    def get_closest_gtruth(self, frame_num):
        total_from_left = self.get_nearest(frame_num, self.left_to_right, self.last_left_before_reset)
        total_from_right = self.get_nearest(frame_num, self.right_to_left, self.last_right_before_reset)
        return total_from_left, total_from_right

    def get_nearest(self, frame_num, seq, start_at):
        total_from_last_reset = 0
        for i in range(start_at, len(seq)-1):
            total_from_last_reset += seq[i][1]
            if seq[i][0] <= frame_num <= seq[i+1][0]:
                return seq[i][0], total_from_last_reset
                # first = num - seq[i][0]
                # second = seq[i+1][0] - num
                # if first < second:
                #     return seq[i]
                # else:
                #     return seq[i+1]

    def reset(self, frame_num):
        # Find last from_left index before reset
        for i in range(len(self.left_to_right)):
            num = self.left_to_right[i][0]
            if frame_num > num:
                self.last_left_before_reset = i-1
        # Find last from_right index before reset
        for i in range(len(self.right_to_left)):
            num = self.right_to_left[i][0]
            if frame_num > num:
                self.last_right_before_reset = i-1



def rewind(buffer, delay):
    """
    Show replay of the last n frames (stored in the buffer).
    :param buffer: a sequence of the frames to be shown
    :param delay: wait for this of milliseconds before showing next frame.
    """
    for frame in buffer:
        cv2.imshow('video', frame)
        cv2.waitKey(delay)


def skip_frames(cap, n, background_subtractor=None, box=None):
    """
    Skip ahead by n frames. The frames are grabbed one by one, but they are not retrieved.
    Raises NoFrameException if a frame can't be grabbed.
    :param cap: VideoCapture object from which frames are skipped
    :param n: number of frames to skip
    :param background_subtractor: if a background subtractor object is given,
     apply() will be called on each skipped frame.
    :param box: required if background_subtractor is given. It specifies how to crop the frame.
     The format should be ((min_x, min_y), (max_x, max_y))
    :returns number of frames skipped (always n, or an exception would have been raised)
    """
    if background_subtractor is not None:
        min_point, max_point = box
        min_x, min_y = min_point
        max_x, max_y = max_point
    for i in range(n):
        if background_subtractor is not None:
            frame = read_frame(cap)
            frame = frame[min_y:max_y, min_x:max_x]
            background_subtractor.apply(frame)
        else:
            ok = cap.grab()
            if not ok:
                raise NoFrameException
    return n


def analyse(config):
    print("Analysing...")
    # Open region file.
    regmap = cv2.imread(config.region_map_path, cv2.IMREAD_GRAYSCALE)
    region_map = RegionMap(regmap)
    config.box = region_map.bounding_box
    print("Frame cropped to "+str(region_map.bounding_box))
    min_point, max_point = region_map.bounding_box
    min_x, min_y = min_point
    max_x, max_y = max_point
    regmap = regmap[min_y:max_y, min_x:max_x]
    region_map = RegionMap(regmap)

    # Open video and background.
    cap = cv2.VideoCapture(config.video_path)
    background = cv2.imread(config.background_path)
    background = background[min_y:max_y, min_x:max_x]

    # Initialize background subtractor
    bgs = BackgroundSubtractorAdaptiveMedian(12, 45, background, region_map.foreground, region_map)
    post_processor = PostProcessor(region_map)

    # Create area : people count function by fitting ground truth data
    get_people_count = FittingPoly(config.pix_people_path, 1)

    # Section border colors to be drawn at each frame.
    region_colors = get_region_colors(region_map)

    # This is the information we actually want:
    # how many poeple crossed the bridge in each direction
    tot_people_entered = [0, 0]

    # old_n is the previous directed count in each region, that is how many people are
    # walking in direction 0 and how many are walking in direction 1
    # It is manually set at the start and has to be manually reset periodically.
    old_n = config.init_directed_count

    # Buffer to store the last n frames for rewinding
    rewind_buffer = deque(maxlen=30)

    if config.reset_path:
        reset_table = ResetTable(config.reset_path, region_map,
                                 config.reset_period, config.start_at)
    if config.tot_count_gtruth_path:
        ground_truth = GroundTruth(config.tot_count_gtruth_path, config.start_at)
    if config.results_path:
        result_writer = ResultWriter(config.results_path, with_gtruth=config.tot_count_gtruth_path)
        # Set first row to 0 (at  time 0, 0 people have been counted)
        if config.tot_count_gtruth_path:
            result_writer.write(config.start_at, config.start_at, config.start_at, 0, 0, 0, 0)
        else:
            result_writer.write(config.start_at, 0, 0)

    averaging_interval = config.averaging_interval
    time_interval = config.frame_interval
    frame_num = -1
    try:
        # Start at parameter means that we must skip frames until we reach the desired point.
        skip_start = config.start_at
        # The initial directed count was given by user, so we skip the first frames.
        skip_first_interval = time_interval - int(averaging_interval / 2)
        frame_num += skip_frames(cap, skip_start + skip_first_interval, bgs, config.box)

        maxframe = config.stop_at
        while frame_num < maxframe:
            # Compute average pixel count and convert to number of people
            # store sum of pixel counts for each frame in the averaging_interval
            pix_count_sum = [0]*len(region_map.regions)
            for i in range(averaging_interval):
                frame_num += 1
                frame = read_frame(cap)
                frame = frame[min_y:max_y, min_x:max_x]
                rewind_buffer.append(frame)  # store frame for use in rewind

                # Extract foreground
                fg_mask = bgs.apply(frame)
                fg_mask = post_processor.process(fg_mask)

                # Estimate people count in region based on number of foreground pixels.
                for i in range(len(region_map.regions)):
                    region = region_map.regions[i]
                    pix_count_sum[i] += np.count_nonzero(fg_mask[region])
            reference_frame = frame_num - int(averaging_interval / 2)

            # average and compute people count
            people_count = []
            for i in range(len(pix_count_sum)):
                pix_count_average = pix_count_sum[i] / averaging_interval
                people_count.append(get_people_count(pix_count_average, i))

            # Solve linear system to get entering flow
            p = config.p
            n_regions = len(region_map.regions)
            b = [0] * (2 * n_regions)
            b[0] = old_n[0][0] * (1 - p[0])
            b[2*n_regions-1] = old_n[-1][1] * (1 - p[1])

            #  Known terms for direction 0
            for i in range(1, n_regions):
                b[i] = old_n[i][0]*(1-p[0]) + old_n[i-1][0]*p[0]
            # Known terms for direction 1
            for i in range(n_regions, 2*n_regions-1):
                j = i - n_regions
                b[i] = old_n[j][1]*(1-p[1]) + old_n[j+1][1]*p[1]
            # New people counts are the known terms for the last set of equations
            b += people_count

            # We build the coefficient matrix using this scheme:
            # n[0][0] + n[0][1] + ... + n[-1][0] + n[-1][1] + people_entered[0] + people_entered[1]
            a = [[0.]*(2*n_regions + 2) for i in range(0, 3*n_regions)]
            for i in range(0, n_regions):
                a[i][i*2] = 1.
            for i in range(n_regions, 2*n_regions):
                j = i - n_regions
                a[i][j*2+1] = 1.
            for i in range(2*n_regions, 3*n_regions):
                j = i - 2*n_regions
                a[i][2*j] = 1.
                a[i][2*j+1] = 1.
            a[0][-2] = -1.
            a[2*n_regions-1][-1] = -1.

            if reference_frame >= 27090:
                bula = 1
            # Solve set of equations
            x, res, rank, singular = np.linalg.lstsq(np.array(a), np.array(b))
            negative = x < 0
            x[negative] = 0
            tot_people_entered[0] += x[-2]
            tot_people_entered[1] += x[-1]
            new_people_entered = (x[-2], x[-1])
            old_n = x[:n_regions*2].reshape((n_regions, 2))

            # Debug - show closest real count
            print('frame: %6d, from_left: %8.3f,   from_right: %8.3f' %
                  (reference_frame, new_people_entered[0], new_people_entered[1]))
            if config.results_path:
                if config.tot_count_gtruth_path:
                    from_left, from_right = ground_truth.get_closest_gtruth(reference_frame)
                    result_writer.write(reference_frame, from_left[0], from_right[0],from_left[1],
                                        new_people_entered[0], from_right[1], new_people_entered[1])
                else:
                    result_writer.write(reference_frame, new_people_entered[0], new_people_entered[1])

            # Automatically reset if possible
            if config.reset_path is not None and reference_frame == reset_table.current_frame:
                old_n = reset_table.current_count
                print('Resetting old_n...\n', old_n)
                reset_table.next()
                # tot_people_entered = [0, 0]
                # if config.tot_count_gtruth_path is not None:
                #     ground_truth.reset(reference_frame)

            # Draw frame.
            contours = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            frame = cv2.drawContours(frame, [c for c in contours], -1, (0, 0, 255), 1)
            draw_region_borders(frame, region_map, region_colors)
            # Save time by not showing windows
            # cv2.imshow('fg_mask', fg_mask)
            # cv2.imshow('video', frame)
            # cv2.imshow('background', bgs.get_background_image())
            # k = cv2.waitKey(1)
            k = 0

            # Press spacebar to pause.
            # Press i to manually reset the directed people count.
            # This is number of people that are walking in a specific direction for each region.
            # The format is [count0, count1], [count0, count1], [count0, count1]
            # e.g.: 3, 2; 0, 1; 5, 5; means that 3 people in region 0 are walking from left to right,
            # 2 people in region 0 are walking from right to left, 0 people in region 1 are walking from left to right
            # 1 person in region 1 is walking from right to left, 5 people in region 2 are walking from left to right,
            # 5 people in region 2 are walking from right to left
            if k == 32:  # spacebar
                print('Paused at frame ', frame_num)
                #print('People count: ', *people_count)
                print(tot_people_entered)
                k = cv2.waitKey(-1)
                while k != 32 and k != 27:
                    if k == 114: # r key
                        rewind(rewind_buffer, region_map)
                    k = cv2.waitKey(-1)
                print('Resumed')
            if k == 27:  # Esc
                break

            # Skip frames
            frame_num += skip_frames(cap, time_interval - averaging_interval, bgs, config.box)

    except NoFrameException:
        pass
    finally:
        cap.release()
    cv2.destroyAllWindows()
    print('Stopped at frame ', frame_num)
    return tot_people_entered


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s command config_path' % (sys.argv[0]))
        exit()
    command = sys.argv[1]
    config_path = sys.argv[2]

    config = Config(config_path)

    if command == 'analyse':
        analyse(config)
    elif command == 'play':
        play(config)
    elif command == 'generate-ground-truth':
        generate_ground_truth(config)
    else:
        print('Wrong command!')
