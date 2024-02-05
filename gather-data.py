import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import argparse
import os
import time

import utils
import constants

# https://raw.githubusercontent.com/rcsmit/python_scripts_rcsmit/master/extras/Gal_Gadot_by_Gage_Skidmore_4_5000x5921_annotated_black_letters.jpg

emotions = constants.EMOTIONS
indecies = constants.INDECIES

print(len(indecies))
cartesian_pairs = utils.cartesian_product(indecies, indecies)
print(cartesian_pairs.shape)

output_data_path = "./data/emotion_landmarks.csv"
columns = ["emotion_label"] + [f'[{x[0]}]x[{x[1]}]' for x in cartesian_pairs]
p_df = []
df = pd.DataFrame(columns=columns)
print(len(columns))

if not os.path.exists(output_data_path):
  df.to_csv(output_data_path, mode='w', header=True)

def run(
  emotion="Happy",
  iterations=200
):

  mp_drawing = mp.solutions.drawing_utils #helps draw all the 468 landmarks -----------------------------------------
  mp_drawing_styles = mp.solutions.drawing_styles # styling --------------------------
  mp_face_mesh = mp.solutions.face_mesh #main model import -----------------------------------------

  font = cv2.FONT_HERSHEY_DUPLEX
  color = (100, 250, 25)

  # For webcam input:
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
        if cv2.waitKey(1) & 0xFF == 32: # if ESC is pressed then the window is closed --------------------------
          if len(p_df) < iterations:
            face_width_squared, face_height_squared = utils.get_face_dimensions_squared(all_landmarks, width, height)
            landmarks2 = all_landmarks[indecies]
            distances = utils.calculate_all_distances(landmarks2, width, height, face_width_squared, face_height_squared).tolist()
            row = [emotion] + distances
            p_df.append(row)
            cv2.putText(image, 
                'Gathering data, '+str(len(p_df)), 
                (25, 50), 
                font, 1, 
                color, 
                2, 
                cv2.LINE_4)
            
          else:
            break

      # Flip the image horizontally for a selfie-view display.
      # image = cv2.flip(image, 1)
      cv2.imshow('MediaPipe Face Mesh', image)
      
      if cv2.waitKey(5) & 0xFF == 27: # if ESC is pressed then the window is closed --------------------------
        break
  
  cap.release()
  # writer.release()

  cv2.destroyAllWindows()

  print(len(p_df))
  print("saving distances ...")
  for r in p_df:
    s1 = time.time_ns()
    df_len = len(df)
    df.loc[df_len] = r
    s2 = time.time_ns()

    print("row number:",df_len, "\tdone in:",str((s2-s1)/10**6)+"ms")

  print("Successfully saved.")
  df.to_csv(output_data_path, mode='a', header=False)

  


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--emotion', type=str, default="Happy", help='emotion label')
    parser.add_argument('-i', '--iterations', type=int, default=200, help='number of samples gathered for one emotion')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)