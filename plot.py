
import sys
# import cv2
import csv
# import glob
# import json
import math

# import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt




# JSONファイルを読み込んで、真値のCSVと比較
# 距離の絶対誤差を出力


def main(input_filename):


  fig = plt.figure(figsize=(10,8))
  ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
  ax2 = plt.subplot2grid((2,2), (1,0))
  ax3 = plt.subplot2grid((2,2), (1,1))


  index = 1
  data = np.loadtxt('/data/demo/csv/' + input_filename + '_1.csv',
        delimiter=',', dtype='float32', usecols=[0,1,2])

  # print (data)
  ax1.set_xlabel('Time')
  ax1.set_ylabel('# of Density Point')
  # print(data)
  ax1.plot(data[:,1],data[:,2])



  # time_data = np.loadtxt('csv_f/{:0=3}_1.csv'.format(int(data[i,0])), delimiter=',', dtype='float32', usecols=[0, 1, 2])

  # figure_ = plt.figure(1) 


  place_data = np.loadtxt('/data/demo/csv/' + input_filename + '_2.csv',
               delimiter=',', dtype='float32', usecols=[0, 1, 2, 3, 4, 5])

  interval_dist = 1.0 # 1.0 m 単位でヒートマップ作成
  xlim = 15 # 10 x 10 の範囲
  ylim = 20
  xoff = 0.0 # キャリブレーションエリアからのオフセット
  yoff = 0.0 
  dense_num = [[0] * (xlim + 2) for i in [1] * (ylim + 2)]
  dense_num4 = [[0] * (xlim + 2) for i in [1] * (ylim + 2)]
  for i in range(len(place_data[:,0])):
    x = math.ceil(place_data[i, 4])
    if x <= 0:
      x = 0
    elif x >= xlim+1:
      x = xlim+1

    y = math.ceil(place_data[i, 5])
    if y <= 0:
      y = 0
    elif y >= ylim+1:
      y = ylim+1

    dense_num[y][x] = dense_num[y][x] + 1
    if place_data[i, 3] > 4.0:
      dense_num4[y][x] = dense_num4[y][x] + 1

  # print(dense_num)

  # ヒートマップ表示
  # im1 = ax1.imshow(dense_num,interpolation='nearest',vmin=0,vmax=1,cmap='jet')
  # ax1.set_xlabel('X')
  # ax1.set_ylabel('Y')
  im2 = ax2.imshow(dense_num,interpolation='nearest',cmap='jet')
  ax2.set_xlabel('X')
  ax2.set_ylabel('Y')
  plt.colorbar(im2, ax=ax2)
  im3 = ax3.imshow(dense_num4,interpolation='nearest',cmap='jet')
  ax3.set_xlabel('X')
  ax3.set_ylabel('Y')
  # ax2.invert_yaxis()
  plt.colorbar(im3, ax=ax3)


  print("ALL: %d" % np.sum(dense_num))
  print("Over 4s: %d" % np.sum(dense_num4))
  

  fig.tight_layout()
  plt.show()

# def draw_bg_image():
#   pass

if __name__ == '__main__':
  args = sys.argv
  main(args[1])
  # main()