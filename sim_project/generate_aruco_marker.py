#!/usr/bin/env python3
"""
generate_aruco_marker.py
--------------------------------------------------------------
Generates the ArUco marker texture PNG used by evader_aruco_marker.sdf.snippet.
Run once, then copy the output into your evader model's materials/textures/ dir.
--------------------------------------------------------------
"""
import cv2
from cv2 import aruco

MARKER_ID = 0                 # must match MARKER_ID in aruco_evader_perception.py
DICT = aruco.DICT_4X4_50
SIZE_PX = 600
OUTFILE = "aruco_marker_0.png"

d = aruco.getPredefinedDictionary(DICT)
if hasattr(aruco, "generateImageMarker"):
    img = aruco.generateImageMarker(d, MARKER_ID, SIZE_PX)          # OpenCV >=4.7
else:
    img = aruco.drawMarker(d, MARKER_ID, SIZE_PX)                   # OpenCV <4.7

# Add a white border -- pure-black-to-edge markers are harder to detect
# reliably against a similarly dark background/shadow in simulation.
import numpy as np
border = 60
canvas = np.full((SIZE_PX + 2 * border, SIZE_PX + 2 * border), 255, dtype=np.uint8)
canvas[border:border + SIZE_PX, border:border + SIZE_PX] = img

cv2.imwrite(OUTFILE, canvas)
print(f"Wrote {OUTFILE} ({canvas.shape[1]}x{canvas.shape[0]}). "
      f"Copy this into <evader_model_dir>/materials/textures/")
