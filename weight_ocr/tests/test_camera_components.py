import unittest

import cv2

from camera.usb_camera import USBCamera
from config import CAMERA, MODEL


class CameraComponentTests(unittest.TestCase):
    def test_phase_one_camera_configuration(self):
        self.assertEqual(CAMERA.device, "/dev/video0")
        self.assertEqual((CAMERA.width, CAMERA.height), (2560, 1440))
        self.assertEqual(CAMERA.fourcc, "MJPG")
        self.assertEqual(CAMERA.fps, 30)

    def test_fourcc_decode(self):
        encoded = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        self.assertEqual(USBCamera._decode_fourcc(encoded), "MJPG")

    def test_phase_three_model_configuration(self):
        self.assertTrue(MODEL.onnx_path.is_file())
        self.assertEqual((MODEL.input_width, MODEL.input_height), (512, 512))


if __name__ == "__main__":
    unittest.main()
