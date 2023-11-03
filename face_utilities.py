import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from collections import OrderedDict
import sys

class Face_utilities():
    '''
    This class contains all needed functions to work with faces in a frame
    '''

    def __init__(self, face_width=200):
        self.detector = None

        self.predictor = None
        self.age_net = None
        self.gender_net = None

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
        self.gender_list = ['Male', 'Female']

        self.desiredLeftEye = (0.35, 0.35)
        self.desiredFaceWidth = face_width
        self.desiredFaceHeight = None

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
        # For dlib’s 68-point facial landmark detector:
        self.FACIAL_LANDMARKS_68_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])

        # For dlib’s 5-point facial landmark detector:
        self.FACIAL_LANDMARKS_5_IDXS = OrderedDict([
            ("right_eye", (2, 3)),
            ("left_eye", (0, 1)),
            ("nose", (4))
        ])

        # last params
        self.last_age = None
        self.last_gender = None
        self.last_rects = None
        self.last_shape = None
        self.last_aligned_shape = None

        # FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS

    def face_alignment(self, frame, shape):
        '''
        Align the face by vertical axis

        Args:
            frame (cv2 image): the original frame. In RGB format.
            shape (array): 68 facial landmarks' co-ords in format of of tuples (x,y)

        Outputs:
            aligned_face (cv2 image): face after alignment
        '''
        # face_aligned = self.face_align.align(frame,gray,rects[0]) # align face

        # print("1: aligned_shape_1 ")
        # print(shape)
        # print("---")

        if (len(shape) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = self.FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = self.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = self.FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = self.FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(frame, M, (w, h),
                                      flags=cv2.INTER_CUBIC)

        # print("1: aligned_shape_1 = {}".format(aligned_shape))
        # print(aligned_shape.shape)

        if (len(shape) == 68):
            shape = np.reshape(shape, (68, 1, 2))

            # cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
            # (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
            # cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]),
            # (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)

        else:
            shape = np.reshape(shape, (5, 1, 2))
            # cv2.rectangle(aligned_face, (aligned_shape[0][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
            # (aligned_shape[1][0],aligned_shape[4][1]), (0,255,0), 0)

            # cv2.rectangle(aligned_face, (aligned_shape[2][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
            # (aligned_shape[3][0],aligned_shape[4][1]), (0,255,0), 0)

        aligned_shape = cv2.transform(shape, M)
        aligned_shape = np.squeeze(aligned_shape)

        # print("---")
        # return aligned_face, aligned_shape
        return aligned_face, aligned_shape

    def face_detection(self, frame):
        '''
        Detect faces in a frame

        Args:
            frame (cv2 image): a normal frame grab from camera or video

        Outputs:
            rects (array): detected faces as rectangles
        '''
        if self.detector is None:
            self.detector = dlib.get_frontal_face_detector()

        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # get all faces in the frame
        rects = self.detector(gray, 0)
        # to get the coords from a rect, use: (x, y, w, h) = face_utils.rect_to_bb(rects[0])

        return rects

    def get_landmarks(self, frame):
        '''
        Get all facial landmarks in a face

        Args:
            frame (cv2 image): the original frame. In RGB format.

        Outputs:
            shape (array): facial landmarks' co-ords in format of of tuples (x,y)
        '''
        if self.predictor is None:
            print("[INFO] load facial landmarks model ...")
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            print("[INFO] Load model - DONE!")

        if frame is None:
            return None, None
        # all face will be resized to a fix size, e.g width = 200
        # face = imutils.resize(face, width=200)
        # face must be gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.face_detection(frame)

        if len(rects) < 0 or len(rects) == 0:
            return None, None

        shape = self.predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        # in shape, there are 68 pairs of (x, y) carrying coords of 68 points.
        # to draw landmarks, use: for (x, y) in shape: cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        return shape, rects

    def ROI_extraction(self, face, shape):
        '''
        Extract 2 cheeks as the ROIs

        Args:
            face (cv2 image): face cropped from the original frame. In RGB format.
            shape (array): facial landmarks' co-ords in format of of tuples (x,y)

        Outputs:
            ROI1 (cv2 image): right-cheek pixels
            ROI2 (cv2 image): left-cheek pixels
            ROI3 (cv2 image): forehead pixels
        '''

        ROI1 = face[shape[41][1]:shape[50][1],
               shape[4][0]:shape[12][0]]

        return ROI1

    def facial_landmarks_remap(self, shape):
        '''
        Need to re-arrange some facials landmarks to get correct params for cv2.fillConvexPoly

        Args:
            shape (array): facial landmarks' co-ords in format of of tuples (x,y)

        Outputs:
            remapped_shape (array): facial landmarks after re-arranged
        '''

        remapped_shape = shape.copy()
        # left eye brow
        remapped_shape[17] = shape[26]
        remapped_shape[18] = shape[25]
        remapped_shape[19] = shape[24]
        remapped_shape[20] = shape[23]
        remapped_shape[21] = shape[22]
        # right eye brow
        remapped_shape[22] = shape[21]
        remapped_shape[23] = shape[20]
        remapped_shape[24] = shape[19]
        remapped_shape[25] = shape[18]
        remapped_shape[26] = shape[17]
        # neatening
        remapped_shape[27] = shape[0]

        remapped_shape = cv2.convexHull(shape)
        # to use remapped_shape
        # mask = np.zeros((face_frame.shape[0], face_frame.shape[1])) #create a black rectangle mask with w, h of the face
        # cv2.fillConvexPoly(mask, remapped_shape[0:27], 1) #fill convex to the mask with remapped_shape

        return remapped_shape

    def face_process(self, frame):
        '''
        full process to extract face, ROI but no age and gender detection

        Args:
            frame (cv2 image): input frame

        Outputs:
            rects (array): detected faces as rectangles
            face (cv2 image): face
            shape (array): facial landmarks' co-ords in format of tuples (x,y)
            aligned_face (cv2 image): face after alignment
            aligned_shape (array): facial landmarks' co-ords of the aligned face in format of tuples (x,y)

        '''

        shape, rects = self.get_landmarks(frame)
        if shape is None:
            return None

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])

        face = frame[y:y + h, x:x + w]
        aligned_face, aligned_shape = self.face_alignment(frame, shape)

        return rects, face, shape, aligned_face, aligned_shape










