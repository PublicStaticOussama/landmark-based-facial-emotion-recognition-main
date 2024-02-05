from keras.models import load_model
import numpy as np
import cv2
import os
import argparse
import mediapipe as mp

import utils
import constants



MODEL_PATH = "./models/ANN2.h5"

emotions = constants.EMOTIONS
indecies = constants.INDECIES

def run(model_path = MODEL_PATH):
  model = load_model(model_path)
  print(model.summary())

  # mp_drawing = mp.solutions.drawing_utils # helps draw all the 468 landmarks -----------------------------------------
  # mp_drawing_styles = mp.solutions.drawing_styles # styling --------------------------
  mp_face_mesh = mp.solutions.face_mesh # main model import -----------------------------------------

  font = cv2.FONT_HERSHEY_DUPLEX
  color = (100, 250, 25)

  # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) # Drawing the face mesh annotations on the image --------------------------
  cap = cv2.VideoCapture(0) # opencv function to get frames from web cam --------------------------
  width = int(cap.get(3))
  height = int(cap.get(4))
  # writer = cv2.VideoWriter("./tmp/gather-data.mp4",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width, height))
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh: # initialize the landmark detection model --------------------------
    
    while cap.isOpened(): # loop through frames of the webcam --------------------------
      success, image = cap.read() # get frame  --------------------------
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue


      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert the frame from BGR to RGB(red, blue, green) format --------------------------
      results = face_mesh.process(image) # apply landmark detection on frame --------------------------

      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks: # if face was detected by model --------------------------
        # utils.draw_styled_landmarks(image, mp_drawing, results.multi_face_landmarks, mp_face_mesh, mp_drawing_styles)
        all_landmarks = np.array(results.multi_face_landmarks[0].landmark)
        
        for i in indecies:
          p = all_landmarks[i]
          cv2.circle(image, (int(p.x*width),int(p.y*height)), 2, (0,0,255), 2)

        face_width_squared, face_height_squared = utils.get_face_dimensions_squared(all_landmarks, width, height)
        all_landmarks = all_landmarks[indecies]
        distances = utils.calculate_all_distances(all_landmarks, width, height, face_width_squared, face_height_squared)
        pred = model.predict(np.array([distances]))

        cv2.putText(image, 
                emotions[np.argmax(pred)], 
                (25, 50), 
                font, 1, 
                color, 
                2, 
                cv2.LINE_4)

      # Flip the image horizontally for a selfie-view display.
      # image = cv2.flip(image, 1)
      cv2.imshow('MediaPipe Face Mesh', image)
      
      if cv2.waitKey(5) & 0xFF == 27: # if ESC is pressed then the window is closed --------------------------
        break

  cap.release()
  # writer.release()

  cv2.destroyAllWindows()


def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('-mp', '--model-path', type=str, default=MODEL_PATH, help='the path of the model that will be used to predict the emotions')
  # parser.add_argument('-o', '--ouput-video-path', type=str, default='', help='')

  opt = parser.parse_args()
  return opt


def main(opt):
  print(opt)
  run(**vars(opt))


if __name__ == "__main__":
  opt = parse_opt()
  main(opt)