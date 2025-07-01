import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import shutil
import os
import cv2
import numpy as np
from PIL import Image

import os
import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, '/kaggle/input/footlib-v0/magalib')
import supervision as sv

def add_100_to_inner_keys(data):
    for outer_key in data.keys():
        inner_dict = data[outer_key]
        # ایجاد یک دیکشنری جدید با کلیدهای افزایش یافته
        new_inner_dict = {inner_key + 100: value for inner_key, value in inner_dict.items()}
        # جایگزینی دیکشنری داخلی با دیکشنری جدید
        data[outer_key] = new_inner_dict
    return data
    
def reverse_dict_values_by_order(original_dict):
    keys = list(original_dict.keys())
    values = list(original_dict.values())
    
    # معکوس کردن لیست مقادیر
    reversed_values = values[::-1]
    
    # ایجاد دیکشنری جدید با کلیدهای اصلی و مقادیر معکوس‌شده
    reversed_dict = {keys[i]: reversed_values[i] for i in range(len(keys))}
    
    return reversed_dict

def merge_nested_dicts(a, b):

    for key in b:
        if key in a:
            # اگر کلید در a وجود دارد، دیکشنری داخلی را ادغام کن
            a[key].update(b[key])
        else:
            # اگر کلید وجود ندارد، آن را به a اضافه کن
            a[key] = b[key]
    return a
    
def reverse_and_rename_images(source_dir, output_dir):

    # ایجاد پوشه مقصد در صورت عدم وجود
    os.makedirs(output_dir, exist_ok=True)
    
    # لیست تمام فایل‌های jpg و مرتب‌سازی معکوس
    jpg_files = sorted(
        [f for f in os.listdir(source_dir) if f.lower().endswith(".jpg")],
        reverse=True
    )
    
    # پردازش هر فایل
    for index, filename in enumerate(jpg_files, start=1):
        # ساخت نام جدید با فرمت ۰۰۰۰۱.jpg و ...
        new_name = f"{index:05d}.jpg"
        
        # کپی فایل با نام جدید
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(output_dir, new_name)
        shutil.copy2(src_path, dst_path)
    
    return len(jpg_files)

        
def calculate_iou(mask1, mask2):

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    return intersection / union



def find_empty(det1,det2,qball):
    #det1 = listdetect[-1]
    #det2 = video_segments[87]
    
    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(det1, det2)


    #if qball in det2.keys():
    #    qballx = list(det2.keys())
    #    z = qballx.index(qball)
    #    iou_matrix[:,z]=0.99

    tps = np.where(np.max(iou_matrix,axis=1)<0.15)
    detc = sv.Detections(xyxy=np.asarray([[2,4,3,4]]));c=[]
    for idx in tps[0]:
        c.append(det1.xyxy[idx])

    if len(c)>=1:
        detc.xyxy = np.asarray(c)
        return detc
    else:
        return None

def compute_iou_matrix(det1, det2):
   
    # Get number of objects in each detection
    n_det1 = len(det1.xyxy)
    n_det2 = len(det2)
    
    # Initialize IoU matrix
    iou_matrix = np.zeros((n_det1, n_det2))
    
    # Convert det1 xyxy to binary masks (assuming same image size as det2 masks)
    # Note: This is a simplified approach - in practice you'd need the image dimensions
    for i in range(n_det1):
        x1, y1, x2, y2 = det1.xyxy[i]
        # Create a mask for det1 object (simplified - actual implementation needs image dimensions)
        # This is a placeholder - you'll need to adjust based on your actual mask dimensions
        mask1 = np.zeros_like(next(iter(det2.values()))[0], dtype=bool)
        mask1[y1:y2, x1:x2] = True

        pj=0;
        for j in det2.keys():
            mask2 = det2[j][0]  # Assuming the mask is stored as first element
            iou_matrix[i, pj] = calculate_iou(mask1, mask2);pj+=1;
            
    return iou_matrix
    
    
    
import numpy as np

def compute_iou_x(mask1, mask2):
    """محاسبه IoU بین دو ماسک باینری"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def filter_objects_with_high_iou(video_segments_c,qball,iou_threshold=0.5):
    """
    حذف objectهایی با ID >= 100 اگر در همان فریم، IoU آنها با objectهای دیگر بیش از حد باشد
    """
    for frame_id, objects in video_segments_c.items():
        low_id_objects = {obj_id: np.squeeze(mask) for obj_id, mask in objects.items() if obj_id < 100}
        high_id_objects = {obj_id: np.squeeze(mask) for obj_id, mask in objects.items() if obj_id >= 100}

        to_remove = set()

        for high_id, high_mask in high_id_objects.items():
            for low_id, low_mask in low_id_objects.items():
                iou = compute_iou_x(high_mask, low_mask)
                if iou > iou_threshold and low_id!=qball:
                    to_remove.add(high_id)
                    break  # همین که با یکی برخورد داشت، کافیه

        for obj_id in to_remove:
            del objects[obj_id]

    return video_segments_c

def check_black_region(mask, area_threshold_ratio=0.125, center_dist_threshold=50, border_margin=10):
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

def only_ball(ball,frame):
    x1,y1,x2,y2 = ball.xyxy[0]
    patch = cv2.resize(frame[y1-5:y2+5,x1-5:x2+5,:],(150,150))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv);green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    s = np.where(green_mask > 0, np.clip(s * 3.9, 0, 255), s)
    d = s.copy()
    d[d!=255]=0;
    d[d==255]=1;
    return check_black_region(d, area_threshold_ratio=1/60, center_dist_threshold=30, border_margin=10)

def ball_tr(listball,listframe,tr=0.80):
    
    listball_f =[];g=0;gg=[];
    for ball in listball:
        if len(ball.xyxy)==1:
            if ball.confidence[0]>tr:
                listball_f.append([ball,g])
                gg.append(g)
        g+=1;
    if len(listball_f)<=5:
        g=0;
        for ball in listball:
            if g in gg:
                continue
            if len(ball.xyxy)==1:
                if ball.confidence[0]>0.5:
                    try:
                        if only_ball(ball,frame)==1:
                            listball_f.append([ball,g])
                    except:
                        pass
                    #if len(listball_f)>=7:
                    #    break
            g+=1;
    return listball_f
    
def reverse_dict_values_by_order_new(original_dict,lnf):

    inv = list(range(0,lnf))[::-1]
    newdict={}
    for key in original_dict.keys():
        newdict.update({inv[key]:original_dict[key]})

    return newdict


from scipy.ndimage import label

def keep_largest_area(mask):
    """
    ورودی: یک ماسک باینری ۲ بعدی یا ۳ بعدی (مثلاً (1, H, W))
    خروجی: همان ماسک با فقط بزرگ‌ترین ناحیه (Connected Component)
    """

    # اگر ماسک ۳ بعدی است، به ۲ بعدی تبدیل کن
    if mask.ndim == 3:
        mask = mask[0]

    # تبدیل به باینری
    binary_mask = mask.astype(bool)

    # لیبل‌گذاری نواحی متصل
    labeled_array, num_features = label(binary_mask)

    if num_features == 0:
        # اگر هیچ ناحیه‌ای وجود ندارد، ماسک صفر برگردان
        return np.zeros_like(mask, dtype=np.uint8)

    # محاسبه اندازه هر ناحیه
    sizes = np.bincount(labeled_array.ravel())

    # اندازه‌ی برچسب ۰ (پس‌زمینه) را صفر کن
    sizes[0] = 0

    # پیدا کردن لیبل بزرگ‌ترین ناحیه
    largest_label = sizes.argmax()

    # تولید ماسک جدید فقط برای ناحیه بزرگ‌تر
    largest_area_mask = (labeled_array == largest_label).astype(np.uint8)

    # اگر ورودی ۳ بعدی بود، خروجی را هم به همان شکل برگردان
    return largest_area_mask[np.newaxis, ...]
    
def create_video(video_segments,video_dir,output_video_path,qball,fps,player_balls=None):

    ellipse_palette = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])
    ellipse_annotator = sv.EllipseAnnotator(color=ellipse_palette, thickness=2)
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=12,
        height=10,
        outline_thickness=1
    )


    frame_names = sorted(os.listdir(video_dir))
    sample_frame = Image.open(os.path.join(video_dir, frame_names[0]))
    frame_height, frame_width = sample_frame.size[::-1]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Color generation
    def generate_color(obj_id):
        np.random.seed(obj_id)
        return np.random.randint(0, 255, size=3).tolist()
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_color = (255, 255, 255)
    thickness = 1
    text_position = (0, 0)
    
    # Process each frame
    for idx, frame_name in enumerate(frame_names):
        frame_path = os.path.join(video_dir, frame_name)
        frame = np.array(Image.open(frame_path).convert("RGB"))
    
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Overlay masks
        if idx in video_segments.keys():

          obj_idq = None
          if player_balls is not None:
              if idx in player_balls.keys():
                obj_idq = player_balls[idx]
                          
          for obj_id, mask in video_segments[idx].items():
              color = generate_color(obj_id)
              binary_mask = mask.astype(bool)
              #for c in range(3):
              #    frame[:, :, c] = np.where(binary_mask,
              #                              (0.6 * frame[:, :, c] + 0.4 * color[c]).astype(np.uint8),
              #                              frame[:, :, c])
    
              ys, xs = np.where(binary_mask)
              if len(xs) > 0 and len(ys) > 0:

                 
                  x1 = np.min(xs);y1=np.min(ys);x2 = np.max(xs);y2=np.max(ys);
                  dummy = sv.Detections(xyxy=np.asarray([[0,0,0,0]]))
                  dummy.xyxy=np.asarray([[x1,y1,x2,y2]])
                  dummy.class_id = [0]#[obj_id]

                  if obj_id==qball:
                      frame = triangle_annotator.annotate(frame, dummy)
                  else:
                      frame = ellipse_annotator.annotate(frame, dummy)

                  if obj_idq is not None:
                      if obj_idq==obj_id:
                          dummy.class_id = [1]
                          frame = ellipse_annotator.annotate(frame, dummy)
                          
                  
                  center_x = int(np.mean(xs))
                  center_y = int(np.mean(ys)) - 10  # Adjust to place text slightly above
    
                  # Text and background settings
                  text = f"{obj_id}"
                  font_scale_id = 0.3
                  thickness_id = 1
    
                  # Get text size
                  (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale_id, thickness_id)
    
                  # Background rectangle coordinates
                  rect_top_left = (center_x, center_y - text_height - baseline)
                  rect_bottom_right = (center_x + text_width, center_y + baseline)
    
                  # Draw filled black rectangle as background
                  cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 0, 0), thickness=cv2.FILLED)
    
                  # Draw white text on top
                  cv2.putText(frame, text, (center_x, center_y), font, font_scale_id, (255, 255, 255), thickness_id, cv2.LINE_AA)
    
    
                  # Convert to BGR for OpenCV
                  #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
                  # Overlay frame index
                  cv2.putText(frame, f"Frame: {idx}", text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Write to video
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_video_path}")

def edge_ratio(image, threshold1=100, threshold2=200):
    # اگر تصویر رنگی باشد آن را به خاکستری تبدیل کن
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # اعمال Canny
    edges = cv2.Canny(gray, threshold1, threshold2)

    # تعداد پیکسل‌های سفید
    white_pixels = np.count_nonzero(edges)

    # تعداد کل پیکسل‌ها
    total_pixels = edges.size

    # نسبت
    ratio = white_pixels / total_pixels
    return ratio

def valid_det(qdet,qframe):
    dds=[];
    for dd in qdet.xyxy:
        x1,y1,x2,y2 = dd;          
        if edge_ratio(qframe[y1:y2,x1:x2,:])>=0.05:
            dds.append(dd)
    qdet.xyxy = np.asarray(dds) 
    return qdet

import numpy as np
from scipy.spatial.distance import cdist

def min_distance_between_masks(mask1, mask2):
    """Calculate the minimum distance between two binary masks."""
    y1, x1 = np.where(mask1)
    y2, x2 = np.where(mask2)

    if len(x1) == 0 or len(x2) == 0:
        return np.inf  # One of the masks is empty

    points1 = np.stack([x1, y1], axis=1)
    points2 = np.stack([x2, y2], axis=1)

    dists = cdist(points1, points2)
    return np.min(dists)


def pballs(video_segments,qball):
    player_balls={};
    for key in video_segments.keys():
        if qball in video_segments[key].keys():
            mask_ball = video_segments[key][qball]
            ious=[];jkey=[];dists=[];
            for okey in video_segments[key].keys():
                if okey!=qball:
                    mask_p = video_segments[key][okey]
                    iou = calculate_iou(mask_p, mask_ball)
                    dist = min_distance_between_masks(mask_p, mask_ball)
                    dists.append(dist)
                    ious.append(iou)
                    jkey.append(okey)
            #print(key,np.sort(dists))
            #print('---------')
            
            if np.max(ious)>0.01:
                idx = np.argmax(ious)
                #print(idx,np.max(ious),dists[idx])
                player_ball = jkey[idx]
                player_balls.update({key:player_ball})

            elif np.min(dists)<5:
                #print(np.min(dists))
                idx = np.argmin(dists)
                player_ball = jkey[idx]
                player_balls.update({key:player_ball})
                
    return player_balls
