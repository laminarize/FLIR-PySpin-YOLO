# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import LOGGER, ROOT, is_colab, is_kaggle, ops
from ultralytics.yolo.utils.checks import check_requirements


@dataclass
class SourceTypes:
    webcam: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreams:
    # YOLOv8 streamloader
    def __init__(self, sources='file.streams', imgsz=640, vid_stride=1):
        """Initialize instance variables and check for consistent input stream shapes."""
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.imgsz = imgsz
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n

################################################################################################
        self.system = PySpin.System.GetInstance()
        camera = self.system.GetCameras()[0]
        #Initiate camera object
        camera.Init()

        #Transport layer device nodemap
        s_node_map = camera.GetTLStreamNodeMap()

        #Device nodemap
        nodemap = camera.GetNodeMap()
################################################################################################
################This block activates chunk mode which will be used to grab framecount metadata##

        chunk_mode_active = PySpin.CBooleanPtr(nodemap.GetNode('ChunkModeActive'))

        if PySpin.IsWritable(chunk_mode_active):
            chunk_mode_active.SetValue(True)

        print('Chunk mode activated...')

        # Enable all types of chunk data
        chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode('ChunkSelector'))

        if not PySpin.IsReadable(chunk_selector) or not PySpin.IsWritable(chunk_selector):
            print('Unable to retrieve chunk selector. Aborting...\n')
            return False

        # Retrieve entries
        entries = [PySpin.CEnumEntryPtr(chunk_selector_entry) for chunk_selector_entry in chunk_selector.GetEntries()]

        print('Enabling entries...')

        # Iterate through our list and select each entry node to enable
        for chunk_selector_entry in entries:
            # Go to next node if problem occurs
            if not PySpin.IsReadable(chunk_selector_entry):
                continue

            chunk_selector.SetIntValue(chunk_selector_entry.GetValue())

            chunk_str = '\t {}:'.format(chunk_selector_entry.GetSymbolic())

            # Retrieve corresponding boolean
            chunk_enable = PySpin.CBooleanPtr(nodemap.GetNode('ChunkEnable'))

            # Enable the boolean, thus enabling the corresponding chunk data
            if not PySpin.IsAvailable(chunk_enable):
                print('{} not available'.format(chunk_str))
            elif chunk_enable.GetValue() is True:
                print('{} enabled'.format(chunk_str))
            elif PySpin.IsWritable(chunk_enable):
                chunk_enable.SetValue(True)
                print('{} enabled'.format(chunk_str))
            else:
                print('{} not writable'.format(chunk_str))
##############################################################################################
#########This block manually configures buffer handling - this is key to daemon thread########

        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
            print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            
        handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(handling_mode_entry):
            print('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
        if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(stream_buffer_count_mode):
            print('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')

        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        if not PySpin.IsAvailable(stream_buffer_count_mode_manual) or not PySpin.IsReadable(stream_buffer_count_mode_manual):
            print('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
        print('Stream Buffer Count Mode set to manual...')

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
        if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
            print('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')

        # Display Buffer Info
        print('\nDefault Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
        print('Default Buffer Count: %d' % buffer_count.GetValue())
        print('Maximum Buffer Count: %d' % buffer_count.GetMax())

        handling_mode_entry = handling_mode.GetEntryByName('OldestFirstOverwrite')
        handling_mode.SetIntValue(handling_mode_entry.GetValue())
        
        buffer_count.SetValue(4)

        print('Buffer count now set to: %d' % buffer_count.GetValue())
        print('Now Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
################################################################################################
################This block defines all camera parameters and constants##########################
 
        self.key = "q"
        self.cam_fps=200
        camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
#        camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
#        exposure_time_to_set = 2000
#        camera.ExposureTime.SetValue(exposure_time_to_set)
        camera.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit10)
        camera.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
        camera.Width.SetValue(640)
        camera.Height.SetValue(480)
        camera.AasRoiEnable.SetValue(True)
        camera.OffsetX.SetValue(40)
        camera.OffsetY.SetValue(30)
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(self.cam_fps)
        w = camera.Width.GetValue()
        h = camera.Height.GetValue()
###################################################################################################
##########This is the beginning of the streamloaders Init##########################################     
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            self.fps = float(camera.AcquisitionFrameRate())
            self.frames[i] = max(int(camera.AcquisitionFrameCount.GetValue()), 0) or float('inf')  # infinite stream fallback
            self.imgs[i] = np.random.rand(480, 640, 3)
            camera.BeginAcquisition()
            print(self.imgs[i].shape)
            self.threads[i] = Thread(target=self.update, args=([i, s, camera]), daemon = True)
            LOGGER.info(f'{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.cam_fps:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')  # newline

        # Check for common shapes
        self.bs = self.__len__()

    def update(self, i, s, camera):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        print("MADE IT TO UPDATE")
        while camera.IsInitialized():

            FlirImage = camera.GetNextImage(1000)
            chunk_data = FlirImage.GetChunkData().GetFrameID()
            if chunk_data % self.vid_stride == 0:
                self.imgs[i] = FlirImage.GetNDArray()
                FlirImage.Release()
                print(f'frameID is: {chunk_data}')

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        self.count = -1
        return self


    def __next__(self):
        """Returns source paths, transformed and original images for processing YOLOv5."""
        self.count += 1
        if self.key != "q":
            camera.EndAcquisition()
            print(camera.IsInitialized())
            camera.DeInit()
            del camera
            cam_list = self.system.GetCameras()
            cam_list.Clear()
            self.system.ReleaseInstance()      
            raise StopIteration
        im0 = self.imgs.copy()
        return self.sources, im0, None, ''

    def __len__(self):
        """Return the length of the sources object."""
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadScreenshots:
    # YOLOv8 screenshot dataloader, i.e. `yolo predict source=screen`
    def __init__(self, source, imgsz=640):
        """source = [screen_number left top width height] (pixels)."""
        check_requirements('mss')
        import mss  # noqa

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.imgsz = imgsz
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor['top'] if top is None else (monitor['top'] + top)
        self.left = monitor['left'] if left is None else (monitor['left'] + left)
        self.width = width or monitor['width']
        self.height = height or monitor['height']
        self.monitor = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}

    def __iter__(self):
        """Returns an iterator of the object."""
        return self

    def __next__(self):
        """mss screen capture: get raw pixels from the screen as np array."""
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f'screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: '

        self.frame += 1
        return str(self.screen), im0, None, s  # screen, img, original img, im0s, s


class LoadImages:
    # YOLOv8 image/video dataloader, i.e. `yolo predict source=image.jpg/vid.mp4`
    def __init__(self, path, imgsz=640, vid_stride=1):
        """Initialize the Dataloader and raise FileNotFoundError if file not found."""
        if isinstance(path, str) and Path(path).suffix == '.txt':  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.imgsz = imgsz
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = 1
        if any(videos):
            self.orientation = None  # rotation degrees
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f'No images or videos found in {p}. '
                                    f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}')

    def __iter__(self):
        """Returns an iterator object for VideoStream or ImageFolder."""
        self.count = 0
        return self

    def __next__(self):
        """Return next image, path and metadata from dataset."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            success, im0 = self.cap.retrieve()
            while not success:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                success, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            if im0 is None:
                raise FileNotFoundError(f'Image Not Found {path}')
            s = f'image {self.count}/{self.nf} {path}: '

        return [path], [im0], self.cap, s

    def _new_video(self, path):
        """Create a new video capture object."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        if hasattr(cv2, 'CAP_PROP_ORIENTATION_META'):  # cv2<4.6.0 compatibility
            self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
            # Disable auto-orientation due to known issues in https://github.com/ultralytics/yolov5/issues/8493
            # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    def _cv2_rotate(self, im):
        """Rotate a cv2 video manually."""
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """Returns the number of files in the object."""
        return self.nf  # number of files


class LoadPilAndNumpy:

    def __init__(self, im0, imgsz=640):
        """Initialize PIL and Numpy Dataloader."""
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im) for im in im0]
        self.imgsz = imgsz
        self.mode = 'image'
        # Generate fake paths
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array."""
        assert isinstance(im, (Image.Image, np.ndarray)), f'Expected PIL/np.ndarray image type, but got {type(im)}'
        if isinstance(im, Image.Image):
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        """Returns the length of the 'im0' attribute."""
        return len(self.im0)

    def __next__(self):
        """Returns batch paths, images, processed images, None, ''."""
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, None, ''

    def __iter__(self):
        """Enables iteration for class LoadPilAndNumpy."""
        self.count = 0
        return self


class LoadTensor:

    def __init__(self, imgs) -> None:
        self.im0 = imgs
        self.bs = imgs.shape[0]
        self.mode = 'image'

    def __iter__(self):
        """Returns an iterator object."""
        self.count = 0
        return self

    def __next__(self):
        """Return next item in the iterator."""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return None, self.im0, None, ''  # self.paths, im, self.im0, None, ''

    def __len__(self):
        """Returns the batch size."""
        return self.bs


def autocast_list(source):
    """
    Merges a list of source of different types into a list of numpy arrays or PIL images
    """
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # filename or uri
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im))
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL or np Image
            files.append(im)
        else:
            raise TypeError(f'type {type(im).__name__} is not a supported Ultralytics prediction source type. \n'
                            f'See https://docs.ultralytics.com/modes/predict for supported source types.')

    return files


LOADERS = [LoadStreams, LoadPilAndNumpy, LoadImages, LoadScreenshots]


def get_best_youtube_url(url, use_pafy=True):
    """
    Retrieves the URL of the best quality MP4 video stream from a given YouTube video.

    This function uses the pafy or yt_dlp library to extract the video info from YouTube. It then finds the highest
    quality MP4 format that has video codec but no audio codec, and returns the URL of this video stream.

    Args:
        url (str): The URL of the YouTube video.
        use_pafy (bool): Use the pafy package, default=True, otherwise use yt_dlp package.

    Returns:
        (str): The URL of the best quality MP4 video stream, or None if no suitable stream is found.
    """
    if use_pafy:
        check_requirements(('pafy', 'youtube_dl==2020.12.2'))
        import pafy  # noqa
        return pafy.new(url).getbest(preftype='mp4').url
    else:
        check_requirements('yt-dlp')
        import yt_dlp
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # extract info
        for f in info_dict.get('formats', None):
            if f['vcodec'] != 'none' and f['acodec'] == 'none' and f['ext'] == 'mp4':
                return f.get('url', None)


if __name__ == '__main__':
    img = cv2.imread(str(ROOT / 'assets/bus.jpg'))
    dataset = LoadPilAndNumpy(im0=img)
    for d in dataset:
        print(d[0])
