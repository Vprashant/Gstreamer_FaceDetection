"""
Filename: Gst_faceDetection.py
Description: GStreamer Pipeline for face detection
Author: Prashant Verma
Email: Prashant27050@gmail.com
Version: 0.0.1
Release Date: 
Requiremnts: 
            opencv-python-headless>=4.5.3
            PyGObject>=3.42.0

"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import cv2
import numpy as np
import time
import json

class VideoProcessor:
    """A class to process video frames and perform face detection."""
    
    def __init__(self, video_path):
        """
        Initialize the VideoProcessor.
        Args:video_path (str): Path to the input video file.
        """
        self.video_path = video_path
        Gst.init(None)
        self.pipeline = self._create_pipeline()
        self.detections = []

    def _create_pipeline(self):
        """Create the GStreamer pipeline for reading and processing the video."""

        pipeline_str = f"filesrc location={self.video_path} ! qtdemux ! h264parse ! avdec_h264 ! videoconvert ! appsink name=appsink emit-signals=true"
        return Gst.parse_launch(pipeline_str)

    def process_frame(self, sample):
        """
        Process a single frame from the video.
        Args: sample (Gst.Sample): The sample containing the video frame.
        Returns: Gst.FlowReturn: Status of the frame processing.
        """

        buf = sample.get_buffer()
        caps = sample.get_caps()
        height = caps[0].get_structure(0).get_value('height')
        width = caps[0].get_structure(0).get_value('width')
        
        data = buf.extract_dup(0, buf.get_size())
        image = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)
        
        for idx, (x, y, w, h) in enumerate(faces):
            center_x = x + w // 2
            center_y = y + h // 2
            detection = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tracker_id": f"face_{idx}",
                "x": center_x,
                "y": center_y,
                "in": 0,
                "h": h,
                "w": w
            }
            self.detections.append(detection)
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imshow("Face Detection", image)
        cv2.waitKey(1)
        
        return Gst.FlowReturn.OK

    def detect_faces(self, gray_image):
        """
        Detect faces in a grayscale image using the Haar Cascade classifier.
        Args:gray_image (numpy.ndarray): Grayscale image data.
        Returns: List of tuples: Coordinates of detected faces.
        """
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    def run(self):
        """Run the video processing pipeline and return the JSON detections."""
        appsink = self.pipeline.get_by_name("appsink")
        appsink.set_property("emit-signals", True)
        appsink.connect("new-sample", self.process_frame)
        
        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            loop = GObject.MainLoop()
            loop.run()
        except KeyboardInterrupt:
            self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()
        
        return json.dumps(self.detections, indent=4)

if __name__ == "__main__":
    video_path = "D:\\video\\sample.mp4"
    processor = VideoProcessor(video_path)
    detections_json = processor.run()
    print(detections_json)
