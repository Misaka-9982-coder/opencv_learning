# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import sys
import argparse

import numpy as np
import cv2 as cv

# Check OpenCV version
assert cv.__version__ >= "4.8.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

parser = argparse.ArgumentParser(
    description="SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition (https://ieeexplore.ieee.org/document/9318547)")
parser.add_argument('--person', '-p', type=str,
                    help='Usage: the person you want to recognize')
parser.add_argument('--model', '-m', type=str, default='face_recognition_sface_2021dec.onnx',
                    help='Usage: Set model path, defaults to face_recognition_sface_2021dec.onnx.')
parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0,
                    help='Usage: Distance type. \'0\': cosine, \'1\': norm_l1. Defaults to \'0\'')

args = parser.parse_args()

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None, result=False, name=''):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, '{:.2f}FPS'.format(fps), (0, 30), cv.FONT_HERSHEY_DUPLEX, 1, text_color)

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        if result is not True:
            name = 'Unknown'
        cv.putText(output, '{:s}'.format(name), (bbox[0], bbox[1]-30), cv.FONT_HERSHEY_DUPLEX, 1, text_color)

    return output

def get_feature(img, detector, recognizer):
    detector.setInputSize([img.shape[1], img.shape[0]])
    # 检测人脸
    face = detector.detect(img)[1]
    if face is None:
        return None, None
    # 对齐人脸
    aligned_face = recognizer.alignCrop(img, face[0][:-1])
    # 提取特征
    return recognizer.feature(aligned_face), face

if __name__ == '__main__':
    # Instantiate SFace for face recognition
    recognizer = cv.FaceRecognizerSF.create(args.model, '')
    # Instantiate YuNet for face detection
    detector = cv.FaceDetectorYN.create('./face_detection_yunet_2023mar.onnx', '', [320, 320], 0.9, 0.3)

    name = args.person.split('.')[0]
    img1 = cv.imread(args.person)

    # Extract feature
    face_feature1, _ = get_feature(img1, detector, recognizer)

    cap = cv.VideoCapture(0)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([w, h])

    tickmeter = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('无法获取摄像头！')
            break

        # Inference
        tickmeter.start()
        face_feature2, face = get_feature(frame, detector, recognizer)
        tickmeter.stop()
        

        if face_feature2 is None:
            cv.imshow('Face Recognition Demo', frame)
        else:
            # Match
            results = False
            if args.dis_type == 0:
                cosine_score = recognizer.match(face_feature1, face_feature2, 0)
                result = True if cosine_score >= 0.363 else False
            else:
                l2_distance = recognizer.match(face_feature1, face_feature2, 1)
                result = True if l2_distance <= 1.128 else False
            # Draw results on the input image
            frame = visualize(frame, face, fps=tickmeter.getFPS(), result=result, name=name)
            cv.imshow('Face Recognition Demo', frame)


        tickmeter.reset()

