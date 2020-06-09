import sys
import numpy as np
import cv2
import re
import pickle
import csv
import glob


# # 定数定義
ESC_KEY = 27     # Escキー
INTERVAL= 33     # 待ち時間
RESIZE = False

def main(input_filename):

  resize_rate = 0.5
  video_capture = cv2.VideoCapture(input_filename)
  fps = video_capture.get(cv2.CAP_PROP_FPS)
  lastframe = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
  # lastframe = int(fps * 120.0)

  h = int(video_capture.get(4) * resize_rate)
  w = int(video_capture.get(3) * resize_rate)
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  out = cv2.VideoWriter(re.split('[.]', input_filename)[-2]+'_resized.avi', fourcc, fps, (w,h))


  frame_num = 0
  while(frame_num <= lastframe):
      ret, frame = video_capture.read()  # frame shape 640*480*3
      if ret != True:
          break

      if (RESIZE):
        frame = cv2.resize(frame , (w,h))
      else:
        offset = [0,0]
        frame = frame[offset[0]:offset[0]+h,offset[1]:offset[1]+w]


      # cv2.imshow('test', frame)
      out.write(frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

      frame_num = frame_num + 1

      sys.stdout.write("\r{}/{}" .format(frame_num,int(lastframe)))
      sys.stdout.flush()

  # 後処理
  video_capture.release()
  out.release()
  cv2.destroyAllWindows()





if __name__ == '__main__':
  args = sys.argv
  main(args[1])