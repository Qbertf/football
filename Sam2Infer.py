import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm
import cv2
import os
import subprocess
import matplotlib.pyplot as plt
import glob
import pickle         
import shutil

from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
import trackutils
import gzip

class Sam2Infer:

  def __init__(self,args):
      os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
      if torch.cuda.is_available():
          device = torch.device("cuda")
      elif torch.backends.mps.is_available():
          device = torch.device("mps")
      else:
          device = torch.device("cpu")
      print(f"using device: {device}")
    
      if device.type == "cuda":
          torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
          if torch.cuda.get_device_properties(0).major >= 8:
              torch.backends.cuda.matmul.allow_tf32 = True
              torch.backends.cudnn.allow_tf32 = True
      elif device.type == "mps":
          print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
          )

      self.device = device
      with open(args.detectionfile, 'rb') as f:
          detectiondata = pickle.load(f)

      self.video_dir  = detectiondata['video_dir']
      self.listdetect = detectiondata['listdetect']
      self.listball   = detectiondata['listball']
      self.listpath   = detectiondata['listpath']
      self.listframes = [];
      for framepath in self.listpath:
          self.listframes.append(cv2.imread(framepath,cv2.IMREAD_COLOR))

      self.model_cfg = args.sam2_PathConfig
      self.sam2_checkpoint = args.sam2_PathModel

      print(detectiondata['video_dir'])
      #print(detectiondata['listpath'])

  def run(self):
      
      ## 1. ball tracking
      listball_f,video_segments_brd = self.ball_detection()

      ## 2. player + ball tracking
      best_det = self.listdetect[0];frame_index = 0;
      best_det = trackutils.valid_det(best_det,self.listframes[frame_index])
      video_segments,qball = self.playsam(self.video_dir+'/',best_det,frame_index,listball_f,showfig=False,qball=99)

      ## 3. reverse tracking
      bx = np.sort(list(video_segments.keys()))[-1]
      det1 = self.listdetect[-1];det1 = trackutils.valid_det(det1,self.listframes[-1])
      det2 = video_segments[bx]
      detc = trackutils.find_empty(det1,det2,qball=99)

      if detc is not None:
          trackutils.reverse_and_rename_images(self.video_dir+'/', 'temp_r')
          video_segments_r,_ = self.playsam('temp_r/',detc,0,listball_f=None,showfig=False,qball=99)
          video_segments_r_reversed = trackutils.reverse_dict_values_by_order_new(video_segments_r,len(self.listframes))
          video_segments_r_reversed_n = trackutils.add_100_to_inner_keys(video_segments_r_reversed)
          video_segments = trackutils.merge_nested_dicts(video_segments, video_segments_r_reversed_n)
          video_segments = trackutils.filter_objects_with_high_iou(video_segments,qball)   
          os.system('rm -rf temp_r')

          if len(video_segments_brd)!=0:
              video_segments = trackutils.merge_nested_dicts(video_segments, video_segments_brd)

      video_segments = self.area_verify(video_segments)
      player_balls = trackutils.pballs(video_segments,qball)
      trackutils.create_video(video_segments=video_segments,video_dir=self.video_dir+'/',output_video_path=self.video_dir.replace('/','_')+'.mp4',qball=qball,fps=5,player_balls=player_balls)

      #with open(self.video_dir.replace('/','_')+'_track.pkl', 'wb') as fb:
      #    pickle.dump({'video_dir':self.video_dir,'video_segments':video_segments,'listpath':self.listpath}, fb)
      with gzip.open(self.video_dir.replace('/','_')+'_track.pkl.gz', 'wb') as f:
          pickle.dump({'video_dir':self.video_dir,'video_segments':video_segments,'listpath':self.listpath}, fb)
      
  def area_verify(self,video_segments):
      for key in video_segments.keys():
          for gkey in video_segments[key].keys():
              mask = video_segments[key][gkey][0]
              binary_mask = mask.astype(bool)
              ys, xs = np.where(binary_mask)
              if len(xs) > 0 and len(ys) > 0:
                  x1 = np.min(xs);y1=np.min(ys);x2 = np.max(xs);y2=np.max(ys);
                  area = (x2-x1) * (y2-y1)
                  if area>=3800:
                      zero = np.zeros_like(video_segments[key][gkey])
                      video_segments[key].update({gkey:zero})
      return video_segments
      
  def ball_detection(self):
      
        listball_f =[];
        listball_f = trackutils.ball_tr(self.listball,self.listframes,tr=0.80)
        video_segments_br = None
        video_segments_brd={}
        if len(listball_f)==0:
            listball_f=None
        else:
            trackutils.reverse_and_rename_images(self.video_dir+'/', 'temp_b')
            inv = list(range(0,len(self.listframes)))[::-1]
            listball_finv=[]
            for record in listball_f:
                listball_finv.append([record[0],inv[record[1]]]) 
                
            video_segments_b,qball = self.playsam('temp_b/',None,0,listball_finv,showfig=False,qball=99)
            video_segments_br = trackutils.reverse_dict_values_by_order_new(video_segments_b,len(inv))
            
            for key in video_segments_br.keys():
                if key<listball_f[0][1]:
                    video_segments_brd.update({key:video_segments_br[key]})
    
            os.system('rm -rf temp_b')
            
        return listball_f,video_segments_brd
    
  def build_sam2_video_predictor_naked(self,yaml_path, ckpt_path=None, device="cuda", mode="eval"):
      cfg = OmegaConf.load(yaml_path)
      OmegaConf.resolve(cfg)
      cfg.model._target_ = "sam2.sam2_video_predictor.SAM2VideoPredictor"
      model = instantiate(cfg.model, _recursive_=True)
      if ckpt_path is not None:
          print(f"Loading checkpoint from {ckpt_path} ...")
          state_dict = torch.load(ckpt_path, map_location="cpu")["model"]
          missing_keys, unexpected_keys = model.load_state_dict(state_dict)
          if missing_keys or unexpected_keys:
              print("Missing keys:", missing_keys)
              print("Unexpected keys:", unexpected_keys)
          else:
              print("Checkpoint loaded successfully.")
    
      model = model.to(device)
      if mode == "eval":
          model.eval()    
      return model

  def playsam(self,video_dir,best_det,frame_index,listball_f=None,showfig=True,qball=99):
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        predictor = self.build_sam2_video_predictor_naked(
            yaml_path=self.model_cfg,
            ckpt_path=self.sam2_checkpoint,
            device="cuda"
        )
        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)
                              
        q=0;
        if best_det is not None:
            for box in best_det.xyxy:
                box = np.asarray(box,dtype=np.float32)
                
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                  inference_state=inference_state,
                  frame_idx=frame_index,
                  obj_id=q,
                  box=box,)
                
                if showfig:
                    plt.figure(figsize=(9, 6))
                    plt.title(f"frame {q}")
                    plt.imshow(Image.open(os.path.join(video_dir, frame_names[q])))
                    sam2utils.show_box(box, plt.gca())
                    sam2utils.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
                
                q+=1
    
        if listball_f is not None:
            for balls in listball_f:
                box,fq = balls
                box = np.asarray(box.xyxy[0],dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                  inference_state=inference_state,
                  frame_idx=fq,
                  obj_id=qball,
                  box=box,)
        else:
            q=-1;
            
        video_segments = {}  
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    
        del predictor
        return video_segments,qball
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Sam2.")


    parser.add_argument("--CONF_THRES", type=float, default=0.3, help="Threshold for Confidence.")
    parser.add_argument("--IOU_THRES", type=float, default=0.3, help="Threshold for IoU.")
    parser.add_argument("--NMS_THRES", type=float, default=0.5, help="Threshold for NMS.")
    parser.add_argument("--BALL_THRES", type=float, default=0.8, help="Threshold for BALL.")

    parser.add_argument("--sam2_PathLib", type=str, default='/kaggle/input/sam2lib/mylibs', help="Path for Sam2 Lib.")
    parser.add_argument("--sam2_PathModel", type=str, default='/kaggle/input/footlib-v0/magalib/sam2.1_hiera_large.pt', help="Path for sam2 model.")
    parser.add_argument("--sam2_PathConfig", type=str, default='/kaggle/input/sam2lib/mylibs/sam2/configs/sam2.1/sam2.1_hiera_l.yaml', help="Path for sam2 Config.")

    parser.add_argument("--detectionfile", type=str, default="", help="pikclefile")
    parser.add_argument("--Video", type=str, default='', help="Test and Display on signle image")

    parser.add_argument("--SOURCE_VIDEO_PATH", type=str, default='', help="Video Path")
    parser.add_argument("--Episodes_path", type=str, default='Frames', help="Episodes path")
    parser.add_argument("--Save_path", type=str, default='', help="save path")


    args = parser.parse_args()
    sys.path.append(args.sam2_PathLib)

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import torch
    import torchvision
    import numpy as np
    instance = Sam2Infer(args)
    instance.run()
