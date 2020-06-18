import numpy as np
import cv2
import sys

# 定数定義
ESC_KEY = 27     # Escキー
INTERVAL= 1     # 待ち時間

# # 元ビデオファイル
def main(input_filename):
    FILE_NAME_LEFT = "/data/output/"+input_filename + "_output1.avi" # 左
    FILE_NAME_RIGHT = "/data/output/"+input_filename + "_output2.avi" # 右

    # 元ビデオファイル読み込み
    video_capture_left = cv2.VideoCapture(FILE_NAME_LEFT)
    frame_num = 0
    video_capture_right = cv2.VideoCapture(FILE_NAME_RIGHT)


    # 保存ビデオファイルの準備
    resize_rate = int(video_capture_left.get(4)/video_capture_right.get(4))
    w = int(video_capture_left.get(3)) + int(video_capture_right.get(3)*resize_rate)
    h = int(video_capture_left.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video_capture_left.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("/data/output/"+input_filename + '_fusion.avi', fourcc, fps, (w, h))

    # 変換処理ループ
    while True:
        end_flag, lframe = video_capture_left.read()
        if not end_flag == True:
          break
        end_flag, rframe = video_capture_right.read()
        if not end_flag == True:
          break
        rframe = cv2.resize(rframe, (int(video_capture_right.get(3)*resize_rate), h))
        # フレーム表示

        # # フレーム書き込み
        frame = cv2.hconcat([lframe, rframe])
        out.write(frame)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # cv2.imshow('',frame)

        # 次のフレーム読み込み
        frame_num = frame_num + 1

    # 終了処理
    cv2.destroyAllWindows()
    video_capture_left.release()
    video_capture_right.release()
    out.release()

if __name__ == '__main__':
  args = sys.argv
  main(args[1])