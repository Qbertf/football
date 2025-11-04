from tqdm import tqdm
import numpy as np
import cv2
import re
import os
from skimage.metrics import structural_similarity as ssim
import sys

def base_gray_color(paths,operator_calibration_file_validate,MATCH_PATH,refsImage):
    
    pair_socre={}
    q=0;
    for path in tqdm(paths[:30]):
        query_frame = cv2.imread(path,0)
        r=0;
        for keyref in refsImage.keys():
            ref_image = cv2.cvtColor(refsImage[keyref], cv2.COLOR_BGR2GRAY)
            pair_key = (path,keyref)
            score = np.sum(query_frame - ref_image)                      
            pair_socre.update({pair_key:score})
            r+=1;
        q+=1;

    return pair_socre

def base_some_color(paths,operator_calibration_file_validate,MATCH_PATH,refsImage):
    
    pair_socre={}
    q=0;
    for path in tqdm(paths[:1000]):
        query_frame = cv2.imread(path)
        query_frame = cv2.cvtColor(query_frame, cv2.COLOR_BGR2LAB)[1:3]

        r=0;
        for keyref in refsImage.keys():
            ref_image = cv2.cvtColor(refsImage[keyref], cv2.COLOR_BGR2LAB)[1:3]
            pair_key = (path,keyref)
            score = np.mean(abs(query_frame - ref_image))                  
            pair_socre.update({pair_key:score})
            r+=1;
        q+=1;

    return pair_socre

def base_tm(paths,operator_calibration_file_validate,MATCH_PATH,refsImage,limit=None):

    '''
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
    
    #with open("log_calib.txt", "w") as f:
    #  f.write("")
    log_file = open("log_calib.txt", "w", buffering=1)
    log_file.write("RUNCALIB PAIRSCORE\n")
    
    log_file.write("REF: " + MATCH_PATH + " ---> " + str(len(refsImage)) + " QUERY: " +  str(len(paths))  +"\n")
   
    
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)  # tqdm uses stderr by default
    '''
    
    pair_socre={}
    q=0;
    if limit is not None:
        paths=paths[:limit]

    refsImage = {k: cv2.cvtColor(cv2.resize(v, None, fx=0.4, fy=0.4), cv2.COLOR_BGR2GRAY)
                   for k, v in refsImage.items()}
    for path in tqdm(paths):
        #query_frame = cv2.imread(path,0)
        query_frame = cv2.resize(cv2.imread(path, 0), None, fx=0.4, fy=0.4)

        r=0;
        for keyref in refsImage.keys():
            #ref_image = cv2.cvtColor(refsImage[keyref], cv2.COLOR_BGR2GRAY)
            ref_image = refsImage[keyref]
            pair_key = (path,keyref)
            score = template_matching(query_frame, ref_image)                    
            pair_socre.update({pair_key:score})
            r+=1;
        q+=1;

    return pair_socre

def base_ssim(paths,operator_calibration_file_validate,MATCH_PATH,refsImage):
    
    pair_socre={}
    q=0;
    for path in tqdm(paths[:30]):
        #query_frame = cv2.imread(path,0)
        query_frame = cv2.resize(cv2.imread(path, 0), None, fx=0.5, fy=0.5)
        r=0;
        for keyref in refsImage.keys():
            ref_image = cv2.cvtColor(refsImage[keyref], cv2.COLOR_BGR2GRAY)
            pair_key = (path,keyref)
            score, _ = ssim(ref_image, query_frame, full=True)                    
            pair_socre.update({pair_key:score})
            r+=1;
        q+=1;

    return pair_socre

def base_sift(paths,operator_calibration_file_validate,MATCH_PATH,refsImage):
    
    pair_socre={}
    q=0;
    for path in tqdm(paths[:30]):
        query_frame = cv2.imread(path,0)
        r=0;
        for keyref in refsImage.keys():
            ref_image = cv2.cvtColor(refsImage[keyref], cv2.COLOR_BGR2GRAY)
            pair_key = (path,keyref)
            score = feature_matching_fast(ref_image, query_frame)                  
            pair_socre.update({pair_key:score})
            r+=1;
        q+=1;

    return pair_socre

def feature_matching_fast(img1, img2, max_features=1000, ratio_test=0.75):
    """
    نسخه بهینه‌شده تطابق ویژگی‌ها با سرعت بالاتر
    
    Parameters:
    - img1, img2: تصاویر ورودی
    - max_features: حداکثر تعداد ویژگی‌های استخراج شده
    - ratio_test: آستانه برای فیلتر کردن تطابق‌های ضعیف
    """
    
    # ایجاد detector ORB با محدودیت تعداد ویژگی‌ها
    orb = cv2.ORB_create(nfeatures=max_features)
    
    # یافتن keypoints و descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # اگر تعداد ویژگی‌ها کم باشد، امتیاز صفر برگردان
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return 0.0
    
    # استفاده از FLANN-based matcher برای سرعت بیشتر
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                       table_number=6,
                       key_size=12,
                       multi_probe_level=1)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # تطابق با روش KNN (سرعت بیشتر + دقت بهتر)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # فیلتر کردن تطابق‌های خوب با استفاده از ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)
    
    # محاسبه امتیاز شباهت
    min_features = min(len(kp1), len(kp2))
    if min_features == 0:
        return 0.0
    
    similarity = len(good_matches) / min_features
    return similarity

def feature_matching(img1, img2):
   
    # ایجاد detector ORB
    orb = cv2.ORB_create()
    
    # یافتن keypoints و descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # ایجاد matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # تطابق ویژگی‌ها
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # محاسبه امتیاز شباهت
    similarity = len(matches) / min(len(kp1), len(kp2))
    return similarity

def template_matching(query, ref,resize_factor=0.5):

    #query = cv2.resize(query, None, fx=resize_factor, fy=resize_factor)
    #ref = cv2.resize(ref, None, fx=resize_factor, fy=resize_factor)

    result = cv2.matchTemplate(ref, query, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    return max_val  #最高 میزان شباهت


import cv2
import torch
from tqdm import tqdm

def template_matching_torch(query_tensor, ref_tensor):
    """Match single query/ref pair on GPU, normalized like TM_CCOEFF_NORMED."""
    q = (query_tensor - query_tensor.mean()) / (query_tensor.std() + 1e-6)
    r = (ref_tensor - ref_tensor.mean()) / (ref_tensor.std() + 1e-6)
    result = torch.nn.functional.conv2d(r, q)
    return result.max().item()

def base_tm_gpu(paths, refsImage, resize_factor=0.5, limit=None, device='cuda'):
    """GPU-accelerated template matching for large sets."""
    pair_score = {}
    if limit is not None:
        paths = paths[:limit]

    # ✅ Preprocess refs once (to grayscale + resize + GPU tensor)
    refs_tensors = {}
    for keyref, ref in refsImage.items():
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        if resize_factor != 1.0:
            ref_gray = cv2.resize(ref_gray, None, fx=resize_factor, fy=resize_factor)
        ref_tensor = torch.tensor(ref_gray, dtype=torch.float32, device=device)[None, None]
        refs_tensors[keyref] = ref_tensor

    for path in tqdm(paths):
        query = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if query is None:
            continue
        if resize_factor != 1.0:
            query = cv2.resize(query, None, fx=resize_factor, fy=resize_factor)
        query_tensor = torch.tensor(query, dtype=torch.float32, device=device)[None, None]

        # ✅ Compare with all refs on GPU
        for keyref, ref_tensor in refs_tensors.items():
            score = template_matching_torch(query_tensor, ref_tensor)
            pair_score[(path, keyref)] = score

    return pair_score


def calculate_homography_between_frames(image1,image2):
 
    #image1 = cv2.imread(path1)
    #image2 = cv2.imread(path2)

    new_mask1 = np.zeros_like(image1)[:,:,0]
    new_mask1[:140,:220]=1;new_mask1[:140,1170:]=1;
    new_mask2=new_mask1.copy()
    
    #print('new_mask1',new_mask1.shape)

    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


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

    #if domask==True:
        #if new_mask1 is not None and 

    pts1_coords = npts1[:, 0, :].astype(int)
    pts2_coords = npts2[:, 0, :].astype(int)
    valid_mask1 = new_mask1[pts1_coords[:, 1], pts1_coords[:, 0]] == 0
    valid_mask2 = new_mask2[pts2_coords[:, 1], pts2_coords[:, 0]] == 0  
    valid_mask = valid_mask1 & valid_mask2
    npts1 = npts1[valid_mask]
    npts2 = npts2[valid_mask]

    H, mask = cv2.findHomography(npts1, npts2, cv2.RANSAC, 5)
    H_INV, mask = cv2.findHomography(npts2, npts1, cv2.RANSAC,5)

    return H,H_INV

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

def calculate_homography_between_frames_novel(image1,image2,device,extractor,matcher):
 
    feats0 = extractor.extract(image1.to(device))
    feats1 = extractor.extract(image2.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # تبدیل تصاویر به فرمت مناسب (Channels-Last)
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()
    
    # اگر تصاویر به شکل (C, H, W) باشند، به (H, W, C) تبدیل می‌کنیم
    if image1.shape[0] == 3:  # Channels-First
        image1 = np.transpose(image1, (1, 2, 0))
    if image2.shape[0] == 3:  # Channels-First
        image2 = np.transpose(image2, (1, 2, 0))
    
    # اطمینان از اینکه تصاویر uint8 هستند
    if image1.dtype != np.uint8:
        image1 = (image1 * 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = (image2 * 255).astype(np.uint8)
    
    # اگر تصویر grayscale است، به RGB تبدیل کنید
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    # تعریف ماسک‌ها بر اساس شکل تصویر
    new_mask1 = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.uint8)
    new_mask1[:140, :220] = 1
    new_mask1[:140, 1170:] = 1
    #new_mask1[200:, :] = 1
    new_mask2 = new_mask1.copy()

    # تبدیل keypointها به numpy
    npts0 = m_kpts0.cpu().numpy().astype(np.float32) if hasattr(m_kpts0, 'cpu') else m_kpts0.astype(np.float32)
    npts1 = m_kpts1.cpu().numpy().astype(np.float32) if hasattr(m_kpts1, 'cpu') else m_kpts1.astype(np.float32)

    # فیلتر کردن keypointها بر اساس ماسک
    pts0_coords = npts0.astype(int)
    pts1_coords = npts1.astype(int)

    # اطمینان از اینکه مختصات در محدوده تصویر باشند
    height, width = new_mask1.shape[:2]
    pts0_coords[:, 0] = np.clip(pts0_coords[:, 0], 0, width-1)
    pts0_coords[:, 1] = np.clip(pts0_coords[:, 1], 0, height-1)
    pts1_coords[:, 0] = np.clip(pts1_coords[:, 0], 0, width-1)
    pts1_coords[:, 1] = np.clip(pts1_coords[:, 1], 0, height-1)

    # اعمال ماسک
    valid_mask0 = new_mask1[pts0_coords[:, 1], pts0_coords[:, 0]] == 0
    valid_mask1 = new_mask2[pts1_coords[:, 1], pts1_coords[:, 0]] == 0
    valid_mask = valid_mask0 & valid_mask1

    # فیلتر کردن keypointهای معتبر
    npts0 = npts0[valid_mask]
    npts1 = npts1[valid_mask]

    H, mask = cv2.findHomography(npts0, npts1, cv2.RANSAC, 0.5)
    H_INV, mask = cv2.findHomography(npts1, npts0, cv2.RANSAC,0.5)

    #print('H',H)
    #print('H_INV',H_INV)
    return H,H_INV
    

def calculate_keypoint(qpath,image1,device,extractor,matcher,outpath):

    try:
        feats0 = extractor.extract(image1.to(device))
    
        #feats0, = [
        #    rbd(x) for x in [feats0]
        #]  # remove batch dimension
    
        #kpts0 = feats0#["keypoints"]
    
        savefile = qpath.replace('.jpg','.pkl').split('/')
        savefile = outpath+savefile[-3]+'_'+savefile[-2]+'_'+savefile[-1]
        with open(savefile,'wb') as f:
            pickle.dump(feats0,f)

    except:
        pass
    

def calculate_homography_between_frames_withkey(image1,image2,device,extractor,matcher,feats0,feats1):


    #print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
    
    matches01 = matcher({"image0": feats0, "image1": feats1})

    #print(matches01)


    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    #print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
    
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # تبدیل تصاویر به فرمت مناسب (Channels-Last)
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()
    
    # اگر تصاویر به شکل (C, H, W) باشند، به (H, W, C) تبدیل می‌کنیم
    if image1.shape[0] == 3:  # Channels-First
        image1 = np.transpose(image1, (1, 2, 0))
    if image2.shape[0] == 3:  # Channels-First
        image2 = np.transpose(image2, (1, 2, 0))
    
    # اطمینان از اینکه تصاویر uint8 هستند
    if image1.dtype != np.uint8:
        image1 = (image1 * 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = (image2 * 255).astype(np.uint8)
    
    # اگر تصویر grayscale است، به RGB تبدیل کنید
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    # تعریف ماسک‌ها بر اساس شکل تصویر
    new_mask1 = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.uint8)
    new_mask1[:140, :220] = 1
    new_mask1[:140, 1170:] = 1
    #new_mask1[200:, :] = 1
    new_mask2 = new_mask1.copy()

    # تبدیل keypointها به numpy
    npts0 = m_kpts0.cpu().numpy().astype(np.float32) if hasattr(m_kpts0, 'cpu') else m_kpts0.astype(np.float32)
    npts1 = m_kpts1.cpu().numpy().astype(np.float32) if hasattr(m_kpts1, 'cpu') else m_kpts1.astype(np.float32)

    # فیلتر کردن keypointها بر اساس ماسک
    pts0_coords = npts0.astype(int)
    pts1_coords = npts1.astype(int)

    # اطمینان از اینکه مختصات در محدوده تصویر باشند
    height, width = new_mask1.shape[:2]
    pts0_coords[:, 0] = np.clip(pts0_coords[:, 0], 0, width-1)
    pts0_coords[:, 1] = np.clip(pts0_coords[:, 1], 0, height-1)
    pts1_coords[:, 0] = np.clip(pts1_coords[:, 0], 0, width-1)
    pts1_coords[:, 1] = np.clip(pts1_coords[:, 1], 0, height-1)

    # اعمال ماسک
    valid_mask0 = new_mask1[pts0_coords[:, 1], pts0_coords[:, 0]] == 0
    valid_mask1 = new_mask2[pts1_coords[:, 1], pts1_coords[:, 0]] == 0
    valid_mask = valid_mask0 & valid_mask1

    # فیلتر کردن keypointهای معتبر
    npts0 = npts0[valid_mask]
    npts1 = npts1[valid_mask]

    #distance, angle_deg , _ = calculate_distance_and_angle(npts0, npts1)
    #tps = np.where(np.asarray(distance)<=25)
    #npts0=npts0[tps];npts1=npts1[tps];

    H, mask = cv2.findHomography(npts0, npts1, cv2.RANSAC, 0.5)
    H_INV, mask = cv2.findHomography(npts1, npts0, cv2.RANSAC,0.5)

    #print('H',H)
    #print('H_INV',H_INV)
    return H,H_INV


import cv2
import numpy as np
import math
import pickle
def calculate_distance_and_angle(pts1, pts2):
    """
    Calculate distance and angle between corresponding points
    
    Parameters:
    pts1, pts2: numpy arrays of shape (N, 1, 2) containing corresponding points
    
    Returns:
    distances: list of Euclidean distances between corresponding points
    angles: list of angles in degrees between corresponding points
    vectors: list of displacement vectors (dx, dy)
    """
    
    # Ensure points are in correct shape
    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)
    
    distances = []
    angles = []
    vectors = []
    
    for p1, p2 in zip(pts1, pts2):
        # Calculate displacement vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Calculate Euclidean distance
        dist = math.sqrt(dx**2 + dy**2)
        
        # Calculate angle in degrees (0-360)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad) % 360
        
        distances.append(dist)
        angles.append(angle_deg)
        vectors.append((dx, dy))
    
    return distances, angles, vectors



