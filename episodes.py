import pandas as pd
import numpy as np

def extract_episodes(csv_path, *, tol=1e-3):
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


import os
import subprocess

def extract_frames_from_episodes(
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

