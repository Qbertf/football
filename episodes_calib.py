import pandas as pd
import numpy as np
import csv

def extract_episodes_old(csv_path, *, tol=1e-3):
    """
    csv_path : str  – مسیر فایل CSV
    tol      : float – تلورانس برای مقایسه اعداد اعشاری (پیش‌فرض ۰٫۰۰۰۱)

    خروجی:
        episodes : list[dict] – هر دیکشنری شامل start, end, episode_id, rows
    """
    # در صورت نیاز جداکنندهٔ ستون‌ها را اصلاح کنید
    df = pd.read_csv(csv_path)

    episodes = []
    ep_start = max(df.loc[0, 'Start'], 0)  # اگر Start منفی بود، صفر قرار می‌دهیم
    ep_rows = [0]                       # شماره سطرهای عضو اپیزود فعلی

    for i in range(len(df) - 1):
        this_end = df.loc[i, 'End']
        next_start = max(df.loc[i+1, 'Start'], 0)  # اگر Start منفی بود، صفر قرار می‌دهیم

        # اگر فاصلهٔ زمانی وجود دارد ➜ اپیزود تمام شده است
        if not np.isclose(this_end, next_start, atol=tol):
            episodes.append({
                'episode_id': len(episodes) + 1,
                'start': ep_start,
                'end': this_end,
                'rows': ep_rows.copy(),   # کپی تا لیست بعدی خراب نشود
            })
            # اپیزود جدید
            ep_start = next_start
            ep_rows = [i+1]
        else:
            ep_rows.append(i+1)

    # آخرین اپیزود را اضافه می‌کنیم
    episodes.append({
        'episode_id': len(episodes) + 1,
        'start': ep_start,
        'end': df.loc[len(df)-1, 'End'],
        'rows': ep_rows.copy(),
    })

    return episodes

def extract_episodes(csv_path):
    episodes = []
    data = []

    # خواندن فایل CSV
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # تبدیل Start و End به عدد
            row['Start'] = float(row['Start']) if row['Start'] else None
            row['End'] = float(row['End']) if row['End'] else None
            data.append(row)

    # گروه‌بندی بر اساس EpisodeId
    grouped = {}
    #unq=[]
    for row in data:
        eid = row['EpisodeId']
        #unq.append(eid)
        if eid not in grouped:
            grouped[eid] = []
        grouped[eid].append(row)

    #print('zzzz ',len(unq),len(set(unq)))
    # استخراج اولین Start و آخرین End برای هر EpisodeId
    for i, eid in enumerate(sorted(grouped.keys(), key=lambda x: int(x)), start=1):
        rows = grouped[eid]
        start_time = min(r['Start'] for r in rows if r['Start'] is not None)
        end_time = max(r['End'] for r in rows if r['End'] is not None)

        episodes.append({
            'episode_id': i,          # شماره جدید
            'original_episode_id': eid,  # شماره اصلی
            'start': start_time,
            'end': end_time
        })

    return episodes
    

import os
import subprocess

def extract_frames_from_episodes_old(
    video_path,
    episodes_data,
    output_folder,
    fps=5,
    global_start=None,
    global_end=None,
    verbose=True
):
    """
    Extract frames from video based on episodes data, optionally within a global time range.
    
    Args:
        video_path (str): Path to the input video file.
        episodes_data (list): List of dictionaries containing episode information.
        output_folder (str): Root folder where episode folders will be created.
        fps (int): Frames per second to extract (default: 5).
        global_start (float): Start time for global filtering (optional).
        global_end (float): End time for global filtering (optional).
        verbose (bool): If True, prints progress messages; if False, silent (default: True).
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for episode in episodes_data:
        episode_id = episode['episode_id']
        episode_start = episode['start']
        episode_end = episode['end']

        
        # Skip if episode is entirely outside global range
        if global_start is not None and episode_end < global_start:
            continue
        if global_end is not None and episode_start > global_end:
            continue
        
        # Adjust start and end to fit within global range
        start = max(episode_start, global_start) if global_start is not None else episode_start
        end = min(episode_end, global_end) if global_end is not None else episode_end
        
        # Skip if adjusted duration is <= 0 (no overlap)
        if start >= end:
            continue
        
        # Create episode folder
        episode_folder = os.path.join(output_folder, f"episode_{episode_id}")
        os.makedirs(episode_folder, exist_ok=True)
        
        # FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-ss', str(start),
            '-i', video_path,
            '-t', str(end - start),
            '-vf', f'fps={fps}',
            '-q:v', '2',
            os.path.join(episode_folder, '%05d.jpg')
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=not verbose)
            if verbose:
                print(f"Extracted frames for Episode {episode_id} ({start}s to {end}s)")
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Error in Episode {episode_id}: {e}")

import os
import subprocess
import math
import shutil
import glob
from tqdm import tqdm

from tqdm import tqdm
import sys

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def extract_frames_from_episodes(
    video_path,
    episodes_data,
    output_folder,
    fps=5,
    global_start=None,
    global_end=None,
    max_frames=None,
    exact_epi=None,
    overlap=0,
    recreate=False,
    verbose=True,log=True
):
    os.makedirs(output_folder, exist_ok=True)
    
    if log==True:
      with open("log.txt", "w") as f:
          f.write("")
      log_file = open("log.txt", "w", buffering=1)
      log_file.write("EPISODE\n")
      log_file.write("Number Episodes :" + str(len(episodes_data))+"\n")

      sys.stdout = Tee(sys.stdout, log_file)
      sys.stderr = Tee(sys.stderr, log_file)  # tqdm uses stderr by default
    
    for episode in tqdm(episodes_data):
        episode_id = episode['episode_id']
        episode_start = episode['start']
        episode_end = episode['end']

        
        if exact_epi is not None:
            if exact_epi!=episode_id:
                continue

                
        episode_id_padded = str(episode_id).zfill(4)
        episode_folder = os.path.join(output_folder, f"framebuffer_{episode_id_padded}")
        
        if global_start is not None and episode_end < global_start:
            continue
        if global_end is not None and episode_start > global_end:
            continue
        
        start = max(episode_start, global_start) if global_start is not None else episode_start
        end = min(episode_end, global_end) if global_end is not None else episode_end
        duration = end - start
        
        if duration <= 0:
            continue

        total_frames = int(duration * fps)

        if max_frames is None or total_frames <= max_frames:
            part_folder = os.path.join(episode_folder, "part_0001")
            os.makedirs(part_folder, exist_ok=True)
            _run_ffmpeg(
                video_path,
                start_time=start,
                duration=duration,
                fps=fps,
                output_dir=part_folder,
                start_number=0,
                verbose=verbose
            )
            if recreate:
                _recreate_video_from_frames(part_folder, fps, verbose)
        else:
            step = max_frames - overlap
            num_parts = math.ceil((total_frames - overlap) / step)

            for part_index in range(num_parts):
                part_start_frame = part_index * step
                part_end_frame = min(part_start_frame + max_frames, total_frames)
                
                part_start_time = start + (part_start_frame / fps)
                part_end_time = start + (part_end_frame / fps)
                part_duration = part_end_time - part_start_time

                part_folder = os.path.join(episode_folder, f"part_{str(part_index+1).zfill(4)}")
                os.makedirs(part_folder, exist_ok=True)

                frame_start_number = part_start_frame

                _run_ffmpeg(
                    video_path,
                    start_time=part_start_time,
                    duration=part_duration,
                    fps=fps,
                    output_dir=part_folder,
                    start_number=frame_start_number,
                    verbose=verbose
                )
                if recreate:
                    _recreate_video_from_frames(part_folder, fps, verbose)


def _run_ffmpeg(video_path, start_time, duration, fps, output_dir, start_number, verbose):
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-vf', f'fps={fps}',
        '-start_number', str(start_number + 1),
        '-q:v', '2',
        os.path.join(output_dir, '%05d.jpg')
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=not verbose)
        if verbose:
            print(f"✅ Extracted frames into {output_dir}, starting from frame {start_number + 1}")
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"❌ Error extracting frames in {output_dir}: {e}")


def _recreate_video_from_frames(frames_folder, fps, verbose):
    part_name = os.path.basename(frames_folder)  # مثل part_0001
    parent_folder = os.path.dirname(frames_folder)
    output_video_path = os.path.join(parent_folder, f"{part_name}.mp4")

    # ایجاد یک پوشه موقت برای کپی کردن فریم‌ها با شماره‌گذاری پیوسته از 00001
    temp_folder = os.path.join(frames_folder, "__temp_seq__")
    os.makedirs(temp_folder, exist_ok=True)

    # پیدا کردن تمام فایل‌های فریم
    frames = sorted(glob.glob(os.path.join(frames_folder, '*.jpg')))
    if not frames:
        if verbose:
            print(f"⚠️ No frames found in {frames_folder}")
        return

    # کپی و تغییر نام فریم‌ها به temp_folder
    for idx, frame_path in enumerate(frames):
        new_name = os.path.join(temp_folder, f"{idx+1:05d}.jpg")
        shutil.copy(frame_path, new_name)

    # ساخت ویدیو با فریم‌های مرتب‌شده
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(temp_folder, '%05d.jpg'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_video_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=not verbose)
        if verbose:
            print(f"🎬 Video saved at: {output_video_path}")
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"❌ Error creating video for {frames_folder}: {e}")
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)
