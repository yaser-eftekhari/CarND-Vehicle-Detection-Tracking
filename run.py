import numpy as np
import pickle
from proj5_util import *

# Global parameters
color_space = 'YUV'         # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11                 # HOG orientations
pix_per_cell = 16           # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel = "ALL"         # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)     # Spatial binning dimensions
hist_bins = 128             # Number of histogram bins
spatial_feat = True         # Spatial features on or off
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off
y_start_stop = [358, 650]   # Min and max in y to search in slide_window()
x_start_stop = [576, 1280]  # Min and max in x to search in slide_window()

# Load trained classifier and the normalizer
dist_pickle = pickle.load( open( "./dist_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]

def pipeline(image):
    global hist
    global processed_frame

    draw_image = np.copy(image)

    # Do a search over the entire frame every 5 frames
    if processed_frame % 5 == 0:
        # Find cars with a scale of 1 (on windows of 64x64)
        hot_windows_1 = find_cars(image, y_start_stop, x_start_stop, svc, X_scaler, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, scale=1)

        # Find cars with a scale of 1.5 (on windows of 96x96)
        hot_windows_15 = find_cars(image, y_start_stop, x_start_stop, svc, X_scaler, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, scale=1.5)

        # Combine the detected windows for the two scales
        hot_windows = hot_windows_1 + hot_windows_15

        # Remove false positives
        output_image = hist.remove_duplicate_false_positive(draw_image, hot_windows, 2, color=(255, 0, 0), thick=4)
    # Use previous boundaries for 4 frames out of 5
    else:
        # Look for cars in the previously found boundaries
        hot_windows = search_windows(image, hist.bboxes, svc, X_scaler, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        # Remove false positives
        output_image = hist.remove_duplicate_false_positive(draw_image, hot_windows, 2, color=(255, 0, 0), thick=4)

    processed_frame += 1
    return output_image

from moviepy.editor import VideoFileClip

hist = History(new_frame_factor = 0.4, no_frames = 5)
processed_frame = 0

output = 'project_output_video.mp4'
frames = VideoFileClip("project_video.mp4")

single_frame = frames.fl_image(pipeline)
single_frame.write_videofile(output, audio=False)
