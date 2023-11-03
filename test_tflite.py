import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="tmp/training/test.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
output_data1 = interpreter.get_tensor(output_details[1]['index'])
print(output_data.shape,output_data1.shape)


import numpy as np
import tensorflow as tf
from model_v2 import HR_BR
import cv2
from face_utilities import Face_utilities
from imutils import face_utils




class ColoarMagnify():

    def __init__(self, levels=8, low_freq=0.75, high_freq=4, fps=24):

        self.levels = levels
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fps = fps

    def get_filtered_img(self, src_imgs):

        downSampledFrames = []

        for item in src_imgs:
            pyrmaid = self.buildGaussianPyramid(item, self.levels)
            downSampledFrames.append(pyrmaid)

        concated = self.reshape_concat(downSampledFrames)

        filtered = self.temporalIdealFilter(concated, self.low_freq, self.high_freq, self.fps);

        return filtered

    def buildGaussianPyramid(self, img, levels):

        if (levels < 1):
            raise ("level below 1")
        else:
            result = img
            for l in range(levels):
                down_img = cv2.pyrDown(result)
                result = down_img

        return result

    def temporalIdealFilter(self, src_img, fl, fh, rate):
        filtered = []
        for i in range(3):
            # current and complextI will hold the real and complex value, so we need float

            current = src_img[:, :, i]
            # temp = current
            w = src_img.shape[0]
            h = src_img.shape[1]
            planes = [np.float32(current), np.zeros(shape=(w, h), dtype=np.float32)]

            complexI = cv2.merge(planes)

            temp = cv2.dft(complexI, cv2.DFT_ROWS)

            # accutally, it is a mask
            filter = self.createIdealBandpassFilter(w, h, fl, fh, rate)

            filters = [filter, filter]
            mask = cv2.merge(filters)
            temp = mask * temp

            temp = cv2.idft(temp, cv2.DFT_ROWS)
            planes = cv2.split(temp)

            current = planes[0]
            filtered.append(current)

        result = cv2.merge(filtered)
        result = cv2.normalize(result, 0, 1, cv2.NORM_MINMAX)
        return result

    def createIdealBandpassFilter(self, width, height, fl, fh, rate):
        fl = fl * width / rate
        fh = fh * height / rate
        filter = np.zeros(shape=(width, height), dtype=np.int)
        for i in range(height):
            for j in range(width):

                if (j >= fl and j <= fh):
                    response = 1.0
                else:
                    response = 0.0
                filter[j, i] = response
        return filter

    def reshape_concat(self, frames):

        framew = frames[0].shape[0]
        frameh = frames[0].shape[1]
        new_height = framew * frameh
        length = len(frames)
        new_frame = np.zeros(shape=(new_height, length, 3), dtype=np.int)

        for i, frame in enumerate(frames):
            # reshape image row by row to (*, 3)
            reshaped_img = frame.flatten().reshape(new_height, 3)
            new_frame[:, i] = reshaped_img
        return new_frame

    def deConcat(self, src, frameSize):

        length = src.cols
        result = []
        for i in range(length):
            result.append(src.col[0].reshape([3, frameSize.height]))

        return result

import h5py
def demo():
    fu = Face_utilities()
    update = 3
    fps = 48
    confidences_ = []
    HR = []
    face_input = []
    preprocesing = ColoarMagnify(levels=3)
    with tf.Session() as sess:

        interpreter = tf.lite.Interpreter(model_path="tmp/training/test.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']



        cap = cv2.VideoCapture(0)




        count =0
        text_br=''
        text_hr=''
        while True:
            ret, frame = cap.read()


            if ret:

                ret_process = fu.face_process(frame)

                if ret_process is None:
                    cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #cv2.imshow("frame", frame)

                    #cv2.destroyWindow("face")
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    cv2.destroyAllWindows()
                    #    break
                    continue

                rects, face, shape, aligned_face, aligned_shape = ret_process

                (x, y, w, h) = face_utils.rect_to_bb(rects[0])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # overlay_text = "%s, %s" % (gender, age)
                # cv2.putText(frame, overlay_text ,(x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

                if (len(aligned_shape) == 68):
                    cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[41][1]),
                                  # draw rectangle on right and left cheeks
                                  (aligned_shape[12][0], aligned_shape[50][1]), (0, 255, 0), 0)

                for (x, y) in aligned_shape:
                    cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)

                # for signal_processing
                ROIs = fu.ROI_extraction(aligned_face, aligned_shape)
                if ROIs is None:
                    continue
                faceimg = cv2.resize(ROIs, (64, 48))
                face_input.append(faceimg)

                # Display the resulting frame

                if (len(face_input) == fps):

                    # EVM
                    filtered_img = np.float32(preprocesing.get_filtered_img(face_input))


                    filtered_img = np.expand_dims(filtered_img, 0)
                    interpreter.set_tensor(input_details[0]['index'], filtered_img)

                    interpreter.invoke()
                    #outputs = HB.sess.run(HB.outputs, feed_dict={HB.input_frames: filtered_img})
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    output_data1 = interpreter.get_tensor(output_details[1]['index'])
                    outputs=[output_data,output_data1]



                    y_out = outputs[0] * (240-45 ) + 45
                    br_out=outputs[1]*35+5
                    confidences_.append(y_out)
                    print("BR: ",br_out,'BR_gt: ', " HR: ", y_out, 'HR_gt: ')

                    if (len(confidences_) == update):
                        sum = 0
                        for i in range(update):
                            sum = confidences_[i] + sum
                        confidences = sum / update
                        print("confidences: ", confidences)
                        confidences_ = []
                    face_input=[]
                    count += 1
                    text_br="BR: "+str( br_out[0][0])
                    text_hr="HR: "+str( y_out[0][0])
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
                cv2.putText(frame, text_br, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, text_hr, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow('frame', frame)

                #cv2.imshow("HR_BR", aligned_face)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break







if __name__ == '__main__':

    #main()
    demo()