import os
import sqlite3
import importlib
import reportutils
import create_report
import main
from football import episodes_calib
import glob
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
import pickle
import time
from tqdm import tqdm
import time

def calculate_distance_and_angle(pts1, pts2):
     # --- محاسبه فاصله و زاویه برای هر جفت نقطه متناظر ---
    distances = []
    angles = []
    results = {}  # برای ذخیره نتایج
    
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        # استخراج مختصات
        x1, y1 = pt1[0]
        x2, y2 = pt2[0]
        
        # محاسبه فاصله اقلیدسی
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # محاسبه زاویه (بر حسب درجه)
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # نرمال‌سازی زاویه به بازه 0 تا 360 درجه
        angle_deg = angle_deg % 360
        if angle_deg < 0:
            angle_deg += 360
        # ذخیره نتایج
        distances.append(distance)
        angles.append(angle_deg)
        results[i] = {'distance': distance, 'angle': angle_deg, 'pt1': (x1, y1), 'pt2': (x2, y2)}
    return distances, angles,results
    

def core(output_folder,exact_epi,Match_Path):
    
    folder_part = np.sort(glob.glob(output_folder+'framebuffer_'+str(exact_epi).zfill(4)+'/part_*'));

    magainfo={}

    print('folder_part',folder_part)
    
    for fpart in folder_part:
        paths = np.sort(glob.glob(fpart+'/'+'*.jpg'))

        astart  = paths[0]
        afinish = paths[-1]
        info={}
    
        for i in tqdm(range(0,len(paths)-1)):

          try:
              img1 = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
              img2 = cv2.imread(paths[i+1], cv2.IMREAD_GRAYSCALE)
            
              img1[:140,:]=0;
              img2[:140,:]=0; 
        
              sift = cv2.SIFT_create()
              keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
              keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
        
              bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
              matches = bf.match(descriptors1, descriptors2)
              matches = sorted(matches, key=lambda x: x.distance)
            
              pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
              pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
              distance, angle_deg , _ = calculate_distance_and_angle(pts1, pts2)
        
              tps = np.where(np.asarray(distance)<=50)
              npts1=pts1[tps];npts2=pts2[tps];
        
              H, mask = cv2.findHomography(npts1, npts2, cv2.RANSAC, 5)
              Hv, mask = cv2.findHomography(npts2, npts1, cv2.RANSAC, 5)

              info.update({(i,i+1):[H,Hv]})
          except:
              info.update({(i,i+1):None})
              pass
          #matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=2)
    
          #cv2.imwrite('../jupiter/CALIBx/sadi/'+str(((i))+1).zfill(5)+'.jpg',matched_img)
    
          
        magainfo.update({fpart:[info,astart,afinish]})

    
      #break    
    with open(output_folder+'framebuffer_'+str(exact_epi).zfill(4)+'.pkl','wb') as f:
        pickle.dump(magainfo,f)

    
    
def run(match_id):

    db = main.get_football_db()
    match_id,match_data,host_players,guest_players = create_report.get_information_from_db(db,match_id)
    meta={'match_id':match_id,'match_data':match_data,'guest_players':guest_players,'host_players':host_players}
            
    match_data = meta['match_data']
    folder_name = f"{match_data['host_team_abbreviation']}_{match_data['guest_team_abbreviation']}_{match_data['match_date_hijri'].replace('/', '-')}_{match_data['match_week']}"
    Match_Path= 'Data/'+folder_name+'/'


    epipath = Match_Path+'episode_info.txt'
    with open(epipath,'r') as f:
        epidata = eval(f.read())

    output_folder = '/root/jupiter/TMP_CALIB_'+str(match_id)+'/'
    os.system('mkdir -p '+output_folder)
    fps=25
    max_frames = 875
    
    episodes_data = epidata['episode_data']
    video_path = glob.glob(Match_Path+'full-match/*.mp4')[0]

    for r in episodes_data:
        #print(r['episode_id'])
    
        exact_epi=r['episode_id'];

        #print(output_folder,episodes_data)

        episodes_calib.extract_frames_from_episodes(     video_path,     episodes_data,    \
                                      output_folder,     fps=fps,   \
                                  global_start=None,     global_end=None,     max_frames=max_frames,  \
                                      exact_epi=exact_epi,  \
                                  overlap=0,     recreate=False,     verbose=True,log=True )
    
       
        core(output_folder,exact_epi,Match_Path)
    
        os.system('find '+output_folder+' -mindepth 1 -type d -exec rm -rf {} +')
    
    #MAGA_CALIBRATION = reportutils.grab_calib(Match_Path)
    #MAGA_TRACKING,player_unq = reportutils.grab_track(Match_Path,meta)

    #show_table = reportutils.rendertable(MAGA_TRACKING,MAGA_CALIBRATION,reportutils.sort_unq_players(player_unq),Match_Path)
        
    #return show_table#,MAGA_TRACKING,MAGA_CALIBRATION#,MAGA_TRACKING,MAGA_CALIBRATION,reportutils.sort_unq_players(player_unq),Match_Path