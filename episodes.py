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

def extract_frames_from_episodes(
    video_path,
    episodes_data,
    output_folder,
    fps=5,
    global_start=None,
    global_end=None,
    max_frames=None,
    overlap=0,
    verbose=True
):
    os.makedirs(output_folder, exist_ok=True)
    
    for episode in episodes_data:
        episode_id = episode['episode_id']
        episode_start = episode['start']
        episode_end = episode['end']

        # Pad episode ID (مثلاً 0001)
        episode_id_padded = str(episode_id).zfill(4)
        episode_folder = os.path.join(output_folder, f"episode_{episode_id_padded}")
        
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
            # حتی اگر کم‌تر از max_frames باشد، باز هم part_0001 ایجاد شود
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
        else:
            step = max_frames - overlap
            num_parts = math.ceil((total_frames - overlap) / step)

            for part_index in range(num_parts):
                part_start_frame = part_index * step
                part_end_frame = min(part_start_frame + max_frames, total_frames)
                
                part_start_time = start + (part_start_frame / fps)
                part_end_time = start + (part_end_frame / fps)
                part_duration = part_end_time - part_start_time

                # Pad part index too (مثلاً 0001)
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
