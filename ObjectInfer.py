import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from tqdm import tqdm
import cv2
import os
import subprocess
import matplotlib.pyplot as plt
import glob
import pickle         
import shutil

class ObjectInfer:

  def __init__(self, conf_threshold=0.3, iou_threshold=0.3, nms_threshold=0.5,ball_threshold=0.8,model_path=None):

    self.conf_threshold = conf_threshold
    self.iou_threshold  = iou_threshold
    self.nms_threshold  = nms_threshold

    self.BALL_ID, self.GOALKEEPER_ID, self.PLAYER_ID, self.REFEREE_ID = 0, 1, 2, 3
    self.CLASS2NAME = {self.BALL_ID: 'ball', self.GOALKEEPER_ID: 'goalkeeper', self.PLAYER_ID: 'player', self.REFEREE_ID: 'referee'}

    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
    self.yolo = YOLOv8(model_path, conf_thres=self.conf_threshold, iou_thres=self.iou_threshold)
    self.ellipse_palette = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])
    self.ellipse_annotator = sv.EllipseAnnotator(color=self.ellipse_palette, thickness=2)
    self.ball_threshold = ball_threshold

    print(self.conf_threshold,self.iou_threshold,self.nms_threshold)

    print(self.CLASS2NAME)


  def infer(self,frame: np.ndarray):
    boxes, scores, class_ids = self.yolo(frame)
    if not len(boxes):
        return sv.Detections.empty()
    names = np.array([self.CLASS2NAME[int(cid)] for cid in class_ids])
    return sv.Detections(xyxy=boxes.astype(int), confidence=scores, class_id=class_ids, data={'class_name': names})


  def check_black_region(self,mask, area_threshold_ratio=0.125, center_dist_threshold=50, border_margin=10):
    # اطمینان از اینکه ماسک فقط شامل ۰ و ۲۵۵ است
    mask = (mask * 255).astype(np.uint8) if mask.max() == 1 else mask

    h, w = mask.shape

    # بررسی وجود پیکسل سیاه در مرزهای ۱۰ پیکسلی
    top_border    = mask[0:border_margin, :]
    bottom_border = mask[h-border_margin:h, :]
    left_border   = mask[:, 0:border_margin]
    right_border  = mask[:, w-border_margin:w]

    if np.any(top_border == 0) or np.any(bottom_border == 0) or \
       np.any(left_border == 0) or np.any(right_border == 0):
        return 0

    # معکوس کردن ماسک برای شناسایی نواحی سیاه
    inverted = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # فقط یک ناحیه باید وجود داشته باشه
    if len(contours) != 1:
        return 0

    cnt = contours[0]
    area = cv2.contourArea(cnt)

    image_area = h * w
    area_threshold = area_threshold_ratio * image_area

    if area < area_threshold:
        return 0

    # محاسبه مرکز ناحیه
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return 0
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    image_center = (w // 2, h // 2)
    dist = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)

    if dist > center_dist_threshold:
        return 0

    return 1

  def create_video(self,frames,listball,listdetect,save_path,frame_width,frame_height,fps,show="ball"):
      
      out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
      
      
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.5
      font_color = (0, 0, 0)
      thickness = 1
      text_position = (0, 0)


      confs=[];
      for ball in listball:
          if len(ball.xyxy)==1:
              confs.append(ball.confidence[0])
          else:
              confs.append(0)
        
      if show=="ball":
          qq=0;
          for frame in frames:
              
              canvas = instance.ellipse_annotator.annotate(frame.copy(), listball[qq])
              if confs[qq]!=0:
                  x1,y1,x2,y2 = listball[qq].xyxy[0]
                  cv2.putText(canvas, str(qq)+'_'+str(confs[qq]), (x1,y1), font, font_scale, font_color, thickness, cv2.LINE_AA)
                  #print(canvas[x1:x2,y1:y2,:].shape)
                  #print(canvas[0:x2-x1,0:y2-y1,:].shape)
                  try:
                      patch = cv2.resize(frame[y1-5:y2+5,x1-5:x2+5,:],(150,150))
                      canvas[0:150,0:150,:]=patch 
                      
                      hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                      h, s, v = cv2.split(hsv);green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
                      s = np.where(green_mask > 0, np.clip(s * 3.9, 0, 255), s)
                      
                      canvas[150:300,150:300,0]=s 
                      canvas[150:300,150:300,1]=s 
                      canvas[150:300,150:300,2]=s 
                          
                      
                      d = s.copy()
                      d[d!=255]=0;
                      d[d==255]=1;

                      status = self.check_black_region(d, area_threshold_ratio=1/60, center_dist_threshold=30, border_margin=10)
                      if status==1:
                          canvas[350:500,350:500,0]=d*255 
                          canvas[350:500,350:500,0]=d*255 
                          canvas[350:500,350:500,0]=d*255 
                          
                      
                  except:
                      pass
              out.write(canvas)
              qq+=1;
              
      out.release()
          
        
  def single_video(self,SOURCE_VIDEO_PATH,save_path,fps=5):
      
      listdetect = [];listframes = [];listball = [];
          
      if '.mp4' not in SOURCE_VIDEO_PATH:
          
          trace_frames = np.sort(glob.glob(SOURCE_VIDEO_PATH+'/*.jpg'))
          #frames=[]
          if len(trace_frames)!=0:
              
              frame_height,frame_width,_ = cv2.imread(trace_frames[0]).shape

              for frame_path in tqdm(trace_frames):
                  frame = cv2.imread(frame_path,cv2.IMREAD_COLOR)
                  all_dets = self.infer(frame)
                  ball_det = all_dets[all_dets.class_id == self.BALL_ID]
                  player_track = all_dets[all_dets.class_id != self.BALL_ID].with_nms(self.nms_threshold, class_agnostic=True)
                  listdetect.append(player_track)
                  listframes.append(frame)
                  listball.append(ball_det)
                  #frames.append(frame)
      else:
            
          cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
          frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          fps = int(cap.get(cv2.CAP_PROP_FPS))

          #frames=[]
          while cap.isOpened():
              ret, frame = cap.read()
              if not ret:
                 break
             
              all_dets = self.infer(frame)
              ball_det = all_dets[all_dets.class_id == self.BALL_ID]
              player_track = all_dets[all_dets.class_id != self.BALL_ID].with_nms(self.nms_threshold, class_agnostic=True)
              listdetect.append(player_track)
              listframes.append(frame)
              listball.append(ball_det)
              #frames.append(frame)
        
      with open(SOURCE_VIDEO_PATH.split('/')[-1]+'.pkl', 'wb') as fb:
          pickle.dump({'listdetect':listdetect,'listball':listball,'listframes':listframes}, fb)
  
      if len(save_path)!=0:
          self.create_video(listframes,listball,listdetect,str(save_path),frame_width,frame_height,fps,show="ball")

    
      
      
  def loop_infer(self,SOURCE_VIDEO_PATH,Episodes_path):

      self.FRAMES = Episodes_path

      video_name = SOURCE_VIDEO_PATH.split('/')[-1]
      out_episide =  self.FRAMES+'/'+video_name.replace('.mp4','')

      epi=0;pkldata=[]

      print('out_episide',out_episide,'SOURCE_VIDEO_PATH',SOURCE_VIDEO_PATH)
      for episide_folder in np.sort(glob.glob(out_episide+'/*')):
          parts = np.sort(glob.glob(episide_folder+'/*'))
          ln_parts = len(parts)
          partdetect  = {}
          partframes  = {}
          partpredict = {}
          partpath = {}
          prt = 0

          for episide_folder_part in parts:
              #start_time = time.time()
              listdetect = []
              listframes = [];
              listball = [];
              trace_frames = np.sort(glob.glob(episide_folder_part+'/*.jpg'))
              
              if len(trace_frames)==0:
                  continue
              
              output_video_path = episide_folder_part+'.mp4'

              for frame_path in tqdm(trace_frames):
                  frame = cv2.imread(frame_path,cv2.IMREAD_COLOR)
                  all_dets = self.infer(frame)
                  ball_det = all_dets[all_dets.class_id == self.BALL_ID]
                  player_track = all_dets[all_dets.class_id != self.BALL_ID].with_nms(self.nms_threshold, class_agnostic=True)
                  listdetect.append(player_track)
                  listframes.append(frame)
                  listball.append(ball_det)
                  
                  #print('listball',listball)
               
            ################## BALL ###################

              listball_f =[];g=0;
              for ball in listball:
                  if len(ball.xyxy)==1:
                      if ball.confidence[0]>self.ball_threshold:
                          listball_f.append([ball,g])
                  g+=1;
          
              if len(listball_f)==0:
                  listball_f=None
                    
            ################## BALL ###################

              with open(episide_folder_part+'.pkl', 'wb') as fb:
                  pickle.dump({'video_dir':episide_folder_part,'listdetect':listdetect,'listball':listball,'listball_f':listball_f,'listpath':trace_frames}, fb)
        
              
              pkldata.append(episide_folder_part+'.pkl')
              with open('detection.pkl', 'wb') as fb:
                  pickle.dump(pkldata, fb)
              
              shutil.copy('detection.pkl','detection_c.pkl')
              #print(asd)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Object Detection.")


    parser.add_argument("--CONF_THRES", type=float, default=0.3, help="Threshold for Confidence.")
    parser.add_argument("--IOU_THRES", type=float, default=0.3, help="Threshold for IoU.")
    parser.add_argument("--NMS_THRES", type=float, default=0.5, help="Threshold for NMS.")
    parser.add_argument("--BALL_THRES", type=float, default=0.8, help="Threshold for BALL.")

    parser.add_argument("--Yolo_PathLib", type=str, default='C:/Users/Farnoosh/Desktop/Football/sports/ONNX-YOLOv8-Object-Detection/', help="Path for Yolo Lib.")
    parser.add_argument("--Yolo_PathModel", type=str, default='C:/Users/Farnoosh/Desktop/Football/sports/roboflow/roboflow/football-players-detection-3zvbc/11/weights.onnx', help="Path for Yolo model.")
    parser.add_argument("--Yolo_anotherLib", type=str, default='', help="Path for another Lib.")

    parser.add_argument("--Image", type=str, default="", help="Test and Display on signle image")
    parser.add_argument("--Video", type=str, default='', help="Test and Display on signle image")

    parser.add_argument("--SOURCE_VIDEO_PATH", type=str, default='', help="Video Path")
    parser.add_argument("--Episodes_path", type=str, default='Frames', help="Episodes path")
    parser.add_argument("--Save_path", type=str, default='', help="save path")

    

    
    args = parser.parse_args()
    sys.path.append(args.Yolo_PathLib)
    sys.path.insert(0, args.Yolo_anotherLib)
    from yolov8 import YOLOv8
    import supervision as sv


    instance = ObjectInfer(conf_threshold=args.CONF_THRES,iou_threshold=args.IOU_THRES,nms_threshold=args.NMS_THRES,ball_threshold=args.BALL_THRES,model_path=args.Yolo_PathModel)

    if args.Image!="":
      frame = cv2.imread(args.Image,cv2.IMREAD_COLOR)
      response = instance.infer(frame)
  
      canvas = instance.ellipse_annotator.annotate(frame, response)
      plt.imshow(canvas)
  
      print('args.Video',args.Video)
    if args.Video=='':
        instance.loop_infer(args.SOURCE_VIDEO_PATH,args.Episodes_path)
    else:
        instance.single_video(args.Video,args.Save_path,fps=5)

    #print(response)
