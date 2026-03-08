import os
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp

def align_face(img, kps, output_size=(112,112)):
    """
    Align face using 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
    Args:
        img: np.ndarray HWC
        kps: 5x2 numpy array
        output_size: final aligned face size
    Returns:
        aligned_face: np.ndarray (H,W,3)
    """
    # Canonical 5-point coordinates for AdaFace
    src = np.array([
        [30.2946, 51.6963],   # left eye
        [65.5318, 51.5014],   # right eye
        [48.0252, 71.7366],   # nose
        [33.5493, 92.3655],   # left mouth
        [62.7299, 92.2041]    # right mouth
    ], dtype=np.float32)

    dst = kps.astype(np.float32)

    # If output size width != 112, scale the canonical coordinates
    if output_size[1] == 112:
        pass
    else:
        scale_x = output_size[1] / 96
        scale_y = output_size[0] / 112
        src[:,0] = src[:,0] * scale_x
        src[:,1] = src[:,1] * scale_y

    # Estimate similarity transform
    tfm = estimate_transform('similarity', dst, src)
    aligned = warp(img, inverse_map=tfm.inverse, output_shape=output_size)
    aligned = (aligned * 255).astype(np.uint8)
    return aligned

def detect_faces(img, app, training_phase=True):
    faces = app.get(img)
    if len(faces) == 0:
        return None

    #for training phase ignore if multiple face detected
    if training_phase: 
        if len(faces)>1 or faces[0].kps.shape[0]!=5:
            return None 
        else:
            return faces

    #for testing with multiple face
    valid_faces = []
    face = faces[0]
    for face in faces:
        if face.kps.shape[0] == 5:
            valid_faces.append(face)
    
    return valid_faces
    
    
