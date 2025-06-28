import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as f

from tqdm import tqdm
from PIL import Image
from matplotlib.patches import Polygon

from CalibrationLib.model.cls_hrnet import get_cls_net
from CalibrationLib.model.cls_hrnet_l import get_cls_net as get_cls_net_l

from CalibrationLib.utils.utils_calib import FramebyFrameCalib, pan_tilt_roll_to_orientation
from CalibrationLib.utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict


import CalibUtils
import pickle
import glob
import shutil

class Calibinfer:

  def __init__(self, weights_kp, weights_line,kp_threshold,line_threshold,pnl_refine,device='cuda:0'):
      
      cfg = yaml.safe_load(open("CalibrationLib/config/hrnetv2_w48.yaml", 'r'))
      cfg_l = yaml.safe_load(open("CalibrationLib/config/hrnetv2_w48_l.yaml", 'r'))

      loaded_state = torch.load(weights_kp, map_location=device)
      self.model = get_cls_net(cfg)
      self.model.load_state_dict(loaded_state)
      self.model.to(device)
      self.model.eval()
    
      loaded_state_l = torch.load(weights_line, map_location=device)
      self.model_l = get_cls_net_l(cfg_l)
      self.model_l.load_state_dict(loaded_state_l)
      self.model_l.to(device)
      self.model_l.eval()
    
      self.transform2 = T.Resize((540, 960))
    
      self.kp_threshold = kp_threshold
      self.line_threshold =line_threshold
      self.pnl_refine = pnl_refine
      self.device = device
      
      self.ca = CalibUtils.CalibUtils()



  def inference(self,cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine):
      
      frame = Image.fromarray(frame)
    
      frame = f.to_tensor(frame).float().unsqueeze(0)
      _, _, h_original, w_original = frame.size()
      frame = frame if frame.size()[-1] == 960 else self.transform2(frame)
      frame = frame.to(self.device)
      b, c, h, w = frame.size()
    
      with torch.no_grad():
          heatmaps = model(frame)
          heatmaps_l = model_l(frame)
    
      kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
      line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
      kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
      lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
      kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)
    
      cam.update(kp_dict, lines_dict)
      try:
          final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)
          return final_params_dict
      except:
          return None
        
        
 

        
  def infer(self,cam,frame,returno,indexq,matrix_inv=None):
      
      final_params_dict = self.inference(cam, frame, self.model, self.model_l, self.kp_threshold, self.line_threshold, self.pnl_refine)
      if final_params_dict is not None:
          P = self.ca.projection_from_cam_params(final_params_dict)
          projected_frame,info,Flag,lino = self.ca.project(frame, P,indexq,matrix_inv)
          returno.update({indexq:[P,info,Flag,lino,matrix_inv]})
           
      else:
          projected_frame = frame # If no params, just use original frame
           
          
      return returno,final_params_dict

  
  def single_video(self,SOURCE_VIDEO_PATH,save_path,showinfo=False,fps=5,Enhancement=False):
      
      matrix_inv=None
      if '.mp4' not in SOURCE_VIDEO_PATH:
          
          trace_frames = np.sort(glob.glob(SOURCE_VIDEO_PATH+'/*.jpg'))
          frames=[]
          if len(trace_frames)!=0:
              indexq=-1;returno={}
              frame_height,frame_width,_ = cv2.imread(trace_frames[0]).shape
              cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)

              for frame_path in tqdm(trace_frames):
                  
                  frame = cv2.imread(frame_path,cv2.IMREAD_COLOR)
                  indexq +=1;

                  if Enhancement==True:
                      frame,matrix_inv = self.ca.enhanc1(frame)
                      
                  returno,_ = self.infer(cam,frame,returno,indexq,matrix_inv)
                  frames.append(frame)
          
      else: 
          cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
          frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          fps = int(cap.get(cv2.CAP_PROP_FPS))
          listdetect = [];listframes = [];listball = [];
                  
          cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
    
          indexq=-1;
          returno={}
          frames=[];
          while cap.isOpened():
              ret, frame = cap.read()
              if not ret:
                 break
             
              indexq +=1;
              if Enhancement==True:
                  frame,matrix_inv = self.ca.enhanc1(frame)
                      
              returno,_ = self.infer(cam,frame,returno,indexq,matrix_inv)
              frames.append(frame)
         
      with open(SOURCE_VIDEO_PATH.split('/')[-1]+'_calib.pkl', 'wb') as fb:
          pickle.dump(returno, fb)
  
      if len(save_path)!=0:
          self.ca.create_video(returno,frames,str(save_path),frame_width,frame_height,fps,showinfo=True)

          
      
  def loop_infer(self,SOURCE_VIDEO_PATH,Episodes_path,smooth=False,Enhancement=False):

      self.FRAMES = Episodes_path
      matrix_inv=None

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

              #print(trace_frames)
              frame_height,frame_width,_ = cv2.imread(trace_frames[0]).shape
              #print(frame_width,frame_height)
              cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
              indexq=-1;
              returno={}
      
              all_final_params=[];
              for frame_path in tqdm(trace_frames):
                  frame = cv2.imread(frame_path,cv2.IMREAD_COLOR)
                  indexq +=1;
                  
                  if Enhancement==True:
                      #print(True)
                      frame,matrix_inv = self.ca.enhanc1(frame)

                  returno,final_params_dict = self.infer(cam,frame,returno,indexq,matrix_inv)
                  all_final_params.append(final_params_dict)
                  
                  
              
              
              if smooth==True:
                  indexq=-1;
                  returno={}
                  smoothed_params = self.ca.smooth_params(all_final_params, window_size=3)

                  for frame_path in tqdm(trace_frames):
                      frame = cv2.imread(frame_path,cv2.IMREAD_COLOR)
                      indexq +=1;
                    
                      P = self.ca.projection_from_cam_params(smoothed_params[indexq])
                      projected_frame,info,Flag,lino = self.ca.project(frame.copy(), P,indexq)
                      returno.update({indexq:[P,info,Flag,lino]})
                  
 
              with open(episide_folder_part+'_calib.pkl', 'wb') as fb:
                  pickle.dump({'result':returno,'listpath':trace_frames}, fb)
        
              
              pkldata.append(episide_folder_part+'_calib.pkl')
              with open('calibration.pkl', 'wb') as fb:
                  pickle.dump(pkldata, fb)
              
              shutil.copy('calibration.pkl','calibration_c.pkl')
              
              #print(asd)
    
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Process video or image and plot lines on each frame.")
    parser.add_argument("--weights_kp", type=str, help="Path to the model for keypoint inference.")
    parser.add_argument("--weights_line", type=str, help="Path to the model for line projection.")
    parser.add_argument("--kp_threshold", type=float, default=0.3434, help="Threshold for keypoint detection.")
    parser.add_argument("--line_threshold", type=float, default=0.7867, help="Threshold for line detection.")
    parser.add_argument("--pnl_refine", action="store_true", help="Enable PnL refinement module.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CPU or CUDA device index")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video or image file.")
    parser.add_argument("--input_type", type=str, choices=['video', 'image','episides'], required=True,
                        help="Type of input: 'video' or 'image'.")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the processed video.")
    parser.add_argument("--display", action="store_true", help="Enable real-time display.")
    
    parser.add_argument("--Episodes_path", type=str, default='Frames', help="Episodes path")
    parser.add_argument("--Smooth", type=bool, default=False, help="Smooth")
    parser.add_argument("--Enhancement", type=bool, default=False, help="Smooth")

    

    args = parser.parse_args()



    #def __init__(self, weights_kp, weights_line,kp_threshold,line_threshold,pnl_refine,device='cuda:0'):

    instance = Calibinfer(weights_kp=args.weights_kp,weights_line=args.weights_line,kp_threshold=args.kp_threshold,line_threshold=args.line_threshold,pnl_refine=args.pnl_refine,device=args.device)
    
    if args.input_type=='video':
        instance.single_video(args.input_path,args.save_path,False,5,args.Enhancement)

    elif args.input_type=='episides':
        instance.loop_infer(args.input_path,args.Episodes_path,args.Smooth,args.Enhancement)

