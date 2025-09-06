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

class Sam2Direct:
    
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
      #with open(args.detectionfile, 'rb') as f:
      #    detectiondata = pickle.load(f)

      #self.video_dir  = detectiondata['video_dir']
      #self.listdetect = detectiondata['listdetect']
      #self.listball   = detectiondata['listball']
      #self.listpath   = detectiondata['listpath']
      #self.listframes = [];
      #for framepath in self.listpath:
      #    self.listframes.append(cv2.imread(framepath,cv2.IMREAD_COLOR))

      self.model_cfg = args.sam2_PathConfig
      self.sam2_checkpoint = args.sam2_PathModel
      self.fps = args.fps
      #print(detectiondata['video_dir'])
      #print(detectiondata['listpath'])
      self.Save_Video = args.Save_Video
      self.Save_Segs = args.Save_Segs
      self.Own_ball = args.Own_ball
      
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

      def playsam(self,video_dir,best_det,frame_index,listball_f=None,showfig=True,qball=99,outpath=''):
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
          
        with open(outpath, 'wb') as fb:
            pickle.dump({'video_segments':video_segments,'qball':qball}, fb)

        return video_segments,qball
          
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Sam2.")


    parser.add_argument("--CONF_THRES", type=float, default=0.3, help="Threshold for Confidence.")
    parser.add_argument("--IOU_THRES", type=float, default=0.3, help="Threshold for IoU.")
    parser.add_argument("--NMS_THRES", type=float, default=0.5, help="Threshold for NMS.")
    parser.add_argument("--fps", type=int, default=5, help="Threshold for BALL.")

    parser.add_argument("--sam2_PathLib", type=str, default='/kaggle/input/sam2lib/mylibs', help="Path for Sam2 Lib.")
    parser.add_argument("--sam2_PathModel", type=str, default='/kaggle/input/footlib-v0/magalib/sam2.1_hiera_large.pt', help="Path for sam2 model.")
    parser.add_argument("--sam2_PathConfig", type=str, default='/kaggle/input/sam2lib/mylibs/sam2/configs/sam2.1/sam2.1_hiera_l.yaml', help="Path for sam2 Config.")

    parser.add_argument("--detectionfile", type=str, default="", help="pikclefile")
    parser.add_argument("--Video", type=str, default='', help="Test and Display on signle image")
    
    parser.add_argument("--video_dir", type=str, default='', help="video_dir")
    parser.add_argument("--best_det",  type=str, default='', help="best_det")
    parser.add_argument("--frame_index",  type=int, default='', help="frame_index")
    parser.add_argument("--listball_f",  type=str, default='', help="listball_f")
    parser.add_argument("--outpath",  type=str, default='', help="outpath")

    
    parser.add_argument("--SOURCE_VIDEO_PATH", type=str, default='', help="Video Path")
    parser.add_argument("--Episodes_path", type=str, default='Frames', help="Episodes path")
    parser.add_argument("--Save_path", type=str, default='', help="save path")
    parser.add_argument("--Save_Video", type=str, default='True', help="save path")
    parser.add_argument("--Save_Segs", type=str, default='True', help="save path")
    parser.add_argument("--Own_ball", type=str, default='True', help="save path")

    args = parser.parse_args()
    sys.path.append(args.sam2_PathLib)

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import torch
    import torchvision
    import numpy as np
    instance = Sam2Direct(args)
    instance.run(args.video_dir,eval(args.best_det),args.frame_index,eval(args.listball_f),showfig=True,qball=99,outpath=args.outpath)
