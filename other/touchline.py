import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
#import calib_new


def cal_m(qlines):
    
    x1,y1,x2,y2 = qlines
    if x2 - x1 == 0:  # خط عمودی
        slope = float('inf')  # شیب بی‌نهایت
    else:
        slope = (y2 - y1) / (x2 - x1)

    try:
        return slope.item()
    except:
        return slope
def run(image,vertical=True,tr1=30,tr2=70,tr=50,ang1=75,ang2=110,minLineLength=50,maxLineGap=15):

    hlines=None;hslope=None;vlines=None;vslope=None;
    mask = create_mask(image)

    tps = np.where(mask==1)

    top_y_index = np.argmin(tps[0]); yt = tps[0][top_y_index] ; xt = tps[1][top_y_index]
    bot_x_index = np.argmin(tps[1]); yb = tps[0][bot_x_index] ; xb = tps[1][bot_x_index]
    center_left = (int((xt-xb)//2),yb.item()+int((yb-yt)//2))
    
    top_y_index = np.argmin(tps[0]); yt = tps[0][top_y_index] ; xt = tps[1][top_y_index]
    bot_x_index = np.argmax(tps[1]); yb = tps[0][bot_x_index] ; xb = tps[1][bot_x_index]
    center_right = (xt.item()+int((xb-xt)//2),yb.item()+int((yb-yt)//2))

    sobely = cv2.Sobel(src=mask, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobely[sobely<30]=0;sobely[sobely>=30]=1;
    mask = sobely.copy();

    try:
        imagec,hlines,rawlines,angle = line_h(sobely,image)
    except:
        pass
    if hlines is not None:
        hslope = cal_m(hlines)
    
    if vertical==True:
        vlines = line_v(image,tr1=tr1,tr2=tr2,tr=tr,ang1=ang1,ang2=ang2,minLineLength=minLineLength,maxLineGap=maxLineGap)
        if vlines is not None:
            vslope = cal_m(vlines)
        
    #alllines.append(flines)
    return hlines,hslope,vlines,vslope,center_left,center_right

def line_v(image,tr1=30,tr2=70,tr=50,ang1=75,ang2=110,minLineLength=50,maxLineGap=15):
    
    hsv1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = create_mask(image,fill=False)
    A = mask*hsv1[:,:,1]
    Z = np.zeros_like(A)
    tps = np.where((A>tr1) & (A<tr2))
    Z[tps]=1
    kernel = np.ones((10, 3), np.uint8)
    Z = cv2.dilate(Z, kernel)

    #sobelx = cv2.medianBlur(Z.astype('uint8'), 7)
    lines = cv2.HoughLinesP(Z, 1, np.pi/180, threshold=tr, 
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
    qline=None;
    vertical_lines=[];alllength=[]
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            #print(line[0],abs(angle))
            
            if ang1 < abs(angle) < ang2:  # خطوط تقریباً عمودی
                vertical_lines.append(line[0])
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                alllength.append(length)
    
        if len(alllength)>0:
            line = vertical_lines[np.argmax(alllength)]
            #qline = extend_line_to_image_borders_by_height(line)
            qline = extend_line_to_image_borders(line,image_width=1280)
    if qline is not None:
        return np.asarray(qline[0])
    else:
        return qline
def line_h(sobely,image):
    REF = [0, 0, 1280, 0]
    threshold=100
    min_angle=0; max_angle=5
    min_angle2=180-max_angle; max_angle2=180
    
    edges = sobely.copy().astype('uint8')
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, 
                               minLineLength=50, maxLineGap=10)
        
    # ایجاد کپی از تصویر اصلی برای رسم خطوط
    result_image = image.copy()
    
    if lines is not None:
        filtered_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
    
            # محاسبه شیب
            if x2 - x1 == 0:  # خط عمودی
                slope = float('inf')  # شیب بی‌نهایت
            else:
                slope = (y2 - y1) / (x2 - x1)

        
            flg=0;
            # محاسبه زاویه خط
            #angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            #QUERY = [0, 289, 978, 274]
            
            # محاسبه زاویه
            angle = calculate_angle_between_lines(REF, line[0])

            # نرمال‌سازی زاویه به محدوده 0-180
            #if angle < 0:
            #    angle += 180
    
            #print(angle)
            # بررسی آیا خط در محدوده زاویه‌ای مورد نظر است
            if min_angle <= angle <= max_angle:
                filtered_lines.append((line[0], angle,slope))
                flg=1;
    
            elif min_angle2 <= angle <= max_angle2:
                filtered_lines.append((line[0], angle,slope))
                flg=1
    
            if flg==1:
                # رسم خط روی تصویر
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # نمایش زاویه کنار خط
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(result_image, f"{angle:.1f}°", (mid_x, mid_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        #_, SLOP,SLOPF = find_line_with_min_y_and_slope_sign(filtered_lines)

        res = find_line_with_min_y_and_slope_sign(filtered_lines)
        
        if res is None:
            return None,None,None,None
        else:
            SLOPF= res['result']
            #print(SLOP)
            rline = regression(filtered_lines,SLOP=SLOPF)
            [x1,y1,x2,y2] = rline.astype('int32')
            result_image = cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
            angle = calculate_angle_between_lines(REF, rline.astype('int32'))
            
            '''
            scoreY=[];scoreA=[];scoreS=[];
            for line in filtered_lines:
                x1,y1,x2,y2 = line[0];
                ag = line[1]
                ag = min(180-ag,ag)
                scoreY.append(abs(y2-y1))
                scoreA.append(ag)
                scoreS.append(abs(y2-y1)*ag)
    
            ns = np.argmin(scoreS)
            line_points = filtered_lines[ns][0]
            LINESE,_,_ = extend_line_to_image_borders(line_points, image_width=1280)
            x1,y1,x2,y2 = LINESE;
            result_image = cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            '''
    
            return result_image,rline,filtered_lines,angle
        
def create_mask(image,fill=True):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #lower_green = np.array([35, 40, 40])
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    mask[mask==255]=1;mask[mask!=1]=0;
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    
    # اگر هیچ کامپوننتی پیدا نشد (فقط پس‌زمینه)
    if num_labels <= 1:
        new_mask = np.zeros_like(mask)
    else:
        # پیدا کردن بزرگترین کامپوننت (نادیده گرفتن پس‌زمینه که index=0 است)
        largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        
        # ایجاد ماسک جدید فقط برای بزرگترین ناحیه
        new_mask = np.where(labels == largest_component_idx, 1, 0).astype(np.uint8)

    new_mask = cv2.medianBlur(new_mask, 13)
    if fill==True:
        new_mask = fill_small_holes(new_mask)
    return new_mask


import numpy as np
from scipy import ndimage
from skimage import measure

def fill_small_holes(binary_mask, max_hole_size=2500):


    # 1. حفره‌ها را پیدا کن (نواحی از 0 که کاملاً با 1 احاطه شده‌اند)
    # ابتدا یک ماسک معکوس برای پیدا کردن نواحی پس‌زمینه (background) که شامل حفره‌ها است ایجاد می‌کنیم.
    # فرض می‌کنیم اشیاء ( foreground ) مقدار 1 و پس‌زمینه ( background ) مقدار 0 دارند.

    # ماسک را معکوس کنید تا حفره‌ها (که 0 هستند) به شیء (1) تبدیل شوند.
    inverted_mask = 1 - binary_mask

    # 2. برچسب‌گذاری بر روی ماسک معکوس شده برای پیدا کردن تمام نواحی متصل (connected components)
    # structure پارامتری است که اتصال پیکسل‌ها را تعریف می‌کند (معمولاً 4-همسایگی یا 8-همسایگی)
    structure = np.ones((3, 3), dtype=np.int32)  # 8-همسایگی
    labeled_holes, num_features = ndimage.label(inverted_mask, structure=structure)

    # 3. ویژگی‌های هر ناحیه برچسب خورده (حفره) را دریافت کن
    # properties=['label', 'area']
    region_props = measure.regionprops(labeled_holes)

    # 4. یک کپی از ماسک اصلی ایجاد کن تا تغییرات را روی آن اعمال کنیم
    filled_mask = binary_mask.copy()

    # 5. روی هر ناحیه (حفره) حلقه بزن
    for region in region_props:
        # مختصات پیکسل‌های این ناحیه
        coords = region.coords

        # شرط: اگر مساحت ناحیه (تعداد پیکسل‌ها) کمتر از آستانه باشد
        if region.area < max_hole_size:
            # این ناحیه یک حفره کوچک است. تمام پیکسل‌های آن را در ماسک اصلی به 1 تغییر بده.
            # توجه: از آنجایی که labeled_holes روی ماسک معکوس شده است، این نواحی در ماسک اصلی 0 هستند.
            # coords شامل [row, col] برای هر پیکسل از حفره است.
            filled_mask[coords[:, 0], coords[:, 1]] = 1

    # 6. ماسک پر شده را برگردان
    return filled_mask
    

import numpy as np
from scipy import ndimage

def smooth_lines_with_median_and_ma(lines, kernel_size=3, window_size=3):
    """
    اعمال فیلتر میانه و سپس میانگین متحرک بر روی تمام مختصات خطوط
    با حفظ اولین و آخرین عناصر
    """
    
    # تبدیل به آرایه numpy
    lines_array = np.array(lines)
    
    # استخراج تمام مختصات
    x1_coords = lines_array[:, 0]
    y1_coords = lines_array[:, 1]
    x2_coords = lines_array[:, 2]
    y2_coords = lines_array[:, 3]
    
    # اعمال فیلتر میانه بر روی تمام مختصات
    x1_median = ndimage.median_filter(x1_coords, size=kernel_size)
    y1_median = ndimage.median_filter(y1_coords, size=kernel_size)
    x2_median = ndimage.median_filter(x2_coords, size=kernel_size)
    y2_median = ndimage.median_filter(y2_coords, size=kernel_size)
    
    # تابع برای اعمال میانگین متحرک با حفظ مقادیر ابتدا و انتها
    def moving_average_preserve_ends(data, window_size):
        if len(data) <= window_size:
            return data.copy()
        
        # محاسبه میانگین متحرک برای بخش میانی
        middle_valid = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        # ایجاد آرایه خروجی با اندازه اصلی
        middle_full = np.zeros_like(data, dtype=float)
        
        # محاسبه تعداد عناصر لبه‌ای که باید کپی شوند
        edge_count = (window_size - 1) // 2
        
        # پر کردن بخش میانی با مقادیر کانولوشن
        middle_full[edge_count:-edge_count] = middle_valid
        
        # پر کردن لبه ابتدا با اولین مقدار معتبر
        middle_full[:edge_count] = middle_valid[0]
        
        # پر کردن لبه انتها با آخرین مقدار معتبر  
        middle_full[-edge_count:] = middle_valid[-1]
        
        return middle_full
    
    # اعمال میانگین متحرک با حفظ مقادیر ابتدا و انتها
    x1_smoothed = moving_average_preserve_ends(x1_median, window_size)
    y1_smoothed = moving_average_preserve_ends(y1_median, window_size)
    x2_smoothed = moving_average_preserve_ends(x2_median, window_size)
    y2_smoothed = moving_average_preserve_ends(y2_median, window_size)
    
    # ساخت آرایه نهایی با مختصات هموار شده
    smoothed_lines = []
    for i in range(len(lines)):
        smoothed_line = np.array([
            x1_smoothed[i],     # x1 هموار شده
            y1_smoothed[i],     # y1 هموار شده
            x2_smoothed[i],     # x2 هموار شده
            y2_smoothed[i]      # y2 هموار شده
        ])
        smoothed_lines.append(smoothed_line)
    
    return smoothed_lines


    import numpy as np
from scipy import ndimage

def smooth_lines_with_median_and_ma2(lines, kernel_size=3, window_size=3):
    """
    اعمال فیلتر میانه و سپس میانگین متحرک بر روی مختصات خطوط
    با حفظ مقادیر None و آرایه‌ها و پردازش فقط بر روی مقادیر معتبر
    """
    
    # استخراج تمام مختصات از آرایه‌ها
    x1_coords = []
    y1_coords = []
    x2_coords = []
    y2_coords = []
    valid_indices = []  # اندیس‌هایی که مقدار معتبر دارند
    
    for i, line in enumerate(lines):
        if line is None:
            # برای مقادیر None، مقادیر placeholder اضافه می‌کنیم
            x1_coords.append(None)
            y1_coords.append(None)
            x2_coords.append(None)
            y2_coords.append(None)
        else:
            # برای آرایه‌ها، مقادیر واقعی را استخراج می‌کنیم
            x1_coords.append(line[0])
            y1_coords.append(line[1])
            x2_coords.append(line[2])
            y2_coords.append(line[3])
            valid_indices.append(i)
    
    # تابع برای پردازش یک بعد از داده با حفظ None
    def process_coordinate_with_none(coords, kernel_size, window_size):
        # پیدا کردن اندیس‌های معتبر (غیر None)
        valid_indices = [i for i, val in enumerate(coords) if val is not None]
        
        if not valid_indices:
            return coords.copy()  # همه مقادیر None هستند
        
        # استخراج مقادیر معتبر
        valid_values = [coords[i] for i in valid_indices]
        
        # اعمال فیلتر میانه بر روی مقادیر معتبر
        valid_values_median = ndimage.median_filter(valid_values, size=kernel_size)
        
        # اعمال میانگین متحرک با حفظ لبه‌ها
        if len(valid_values_median) <= window_size:
            smoothed_values = valid_values_median.copy()
        else:
            # محاسبه میانگین متحرک برای بخش میانی
            middle_valid = np.convolve(valid_values_median, np.ones(window_size)/window_size, mode='valid')
            
            # ایجاد آرایه برای مقادیر هموار شده
            smoothed_values = np.zeros_like(valid_values_median, dtype=float)
            
            # محاسبه تعداد عناصر لبه‌ای که باید کپی شوند
            edge_count = (window_size - 1) // 2
            
            # پر کردن بخش میانی با مقادیر کانولوشن
            smoothed_values[edge_count:-edge_count] = middle_valid
            
            # پر کردن لبه ابتدا با اولین مقدار معتبر
            if edge_count > 0:
                smoothed_values[:edge_count] = middle_valid[0]
            
            # پر کردن لبه انتها با آخرین مقدار معتبر  
            if edge_count > 0:
                smoothed_values[-edge_count:] = middle_valid[-1]
            else:
                # اگر window_size زوج باشد
                smoothed_values = middle_valid
        
        # ایجاد آرایه نهایی با همان اندازه ورودی
        result = [None] * len(coords)
        
        # پر کردن مقادیر هموار شده در موقعیت‌های معتبر
        for idx, valid_idx in enumerate(valid_indices):
            result[valid_idx] = smoothed_values[idx]
        
        return result
    
    # پردازش هر مختصات با حفظ None
    x1_smoothed = process_coordinate_with_none(x1_coords, kernel_size, window_size)
    y1_smoothed = process_coordinate_with_none(y1_coords, kernel_size, window_size)
    x2_smoothed = process_coordinate_with_none(x2_coords, kernel_size, window_size)
    y2_smoothed = process_coordinate_with_none(y2_coords, kernel_size, window_size)
    
    # ساخت آرایه نهایی با ساختار اصلی
    smoothed_lines = []
    for i in range(len(lines)):
        if lines[i] is None:
            smoothed_lines.append(None)
        else:
            smoothed_line = np.array([
                x1_smoothed[i],     # x1 هموار شده
                y1_smoothed[i],     # y1 هموار شده
                x2_smoothed[i],     # x2 هموار شده
                y2_smoothed[i]      # y2 هموار شده
            ])
            smoothed_lines.append(smoothed_line)
    
    return smoothed_lines
    
import numpy as np
from sklearn.linear_model import LinearRegression

def regression(data, image_width=1280,SLOP=1):
    all_points = []
    
    # برای هر خط، تمام نقاط روی خط را ایجاد می‌کنیم
    for line, angle, DSLOP  in data:
        
        slope_sign = 1 if DSLOP >= 0 else -1
        if slope_sign!=SLOP:
            continue
        x1, y1, x2, y2 = line
        
        # محاسبه شیب خط (m) و عرض از مبدأ (b)
        if x2 - x1 != 0:  # جلوگیری از تقسیم بر صفر
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            
            # ایجاد نقاط بین دو نقطه
            start_x = min(x1, x2)
            end_x = max(x1, x2)
            
            for x in range(start_x, end_x + 1):
                y = m * x + b
                all_points.append((x, y))
        else:  # خط عمودی
            # برای خطوط عمودی، x ثابت است و y تغییر می‌کند
            x = x1
            start_y = min(y1, y2)
            end_y = max(y1, y2)
            
            for y in range(start_y, end_y + 1):
                all_points.append((x, y))
    
    # اگر هیچ نقطه‌ای وجود ندارد، خط پیش‌فرض برگردانید
    if not all_points:
        return np.array([0, 0, image_width - 1, 0])
    
    # تبدیل به آرایه numpy برای پردازش راحت‌تر
    points_array = np.array(all_points)
    
    # جدا کردن x و y برای رگرسیون
    X = points_array[:, 0].reshape(-1, 1)  # xها به عنوان ویژگی
    y = points_array[:, 1]  # yها به عنوان هدف
    
    # انجام رگرسیون خطی
    model = LinearRegression()
    model.fit(X, y)
    
    # پیش‌بینی y برای ابتدا و انتهای عرض تصویر
    x_min = 0  # شروع از لبه چپ تصویر
    x_max = image_width - 1  # پایان در لبه راست تصویر
    
    y_min_pred = model.predict([[x_min]])[0]
    y_max_pred = model.predict([[x_max]])[0]
    
    # خط فیت شده به صورت (x1, y1, x2, y2) برای کل عرض تصویر
    fitted_line = np.array([x_min, y_min_pred, x_max, y_max_pred])
    return fitted_line

import math

def calculate_angle_between_lines(ref, query):

    # محاسبه شیب خط اول
    dx1 = ref[2] - ref[0]
    dy1 = ref[3] - ref[1]
    
    # محاسبه شیب خط دوم
    dx2 = query[2] - query[0]
    dy2 = query[3] - query[1]
    
    # محاسبه زاویه هر خط با محور x (بر حسب رادیان)
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    
    # محاسبه تفاوت زاویه بین دو خط
    angle_diff = abs(angle1 - angle2)
    
    # تبدیل به درجه و اطمینان از کوچکترین زاویه (بین 0 تا 180 درجه)
    angle_deg = math.degrees(angle_diff)
    
    # اگر زاویه بیشتر از 180 درجه باشد، زاویه مکمل آن را برمی‌گردانیم
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    
    return angle_deg

import numpy as np

def find_line_with_min_y_and_slope_sign(data):
    """
    تحلیل کامل فاصله‌های وزنی بر اساس علامت شیب
    """
    if not data:
        return None
    
    total_positive_distance = 0.0
    total_negative_distance = 0.0
    positive_lines = []
    negative_lines = []
    
    for i, item in enumerate(data):
        x1, y1, x2, y2 = item[0]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if item[2] >= 0:
            total_positive_distance += distance
            positive_lines.append((i, distance, item[2]))
        else:
            total_negative_distance += distance
            negative_lines.append((i, distance, item[2]))
    
    result = 1 if total_positive_distance >= total_negative_distance else -1
    
    return {
        'total_positive': total_positive_distance,
        'total_negative': total_negative_distance,
        'result': result,
        'positive_lines_count': len(positive_lines),
        'negative_lines_count': len(negative_lines),
        'positive_lines_details': positive_lines,
        'negative_lines_details': negative_lines
    }


def add_text_below_image(image, text, background_color=(0, 0, 0), text_color=(255, 255, 255)):
    
    # دریافت ابعاد تصویر اصلی
    height, width = image.shape[:2]
    
    # تعیین اندازه فونت و ضخامت متن
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # محاسبه اندازه متن
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # ایجاد ناحیه سیاه برای متن (20 پیکسل بیشتر از ارتفاع متن برای حاشیه)
    text_area_height = text_height + 40
    
    # ایجاد تصویر سیاه برای ناحیه متن
    text_area = np.zeros((text_area_height, width, 3), dtype=np.uint8)
    text_area[:] = background_color
    
    # قرار دادن متن در مرکز ناحیه سیاه
    text_x = (width - text_width) // 2
    text_y = (text_area_height + text_height) // 2
    
    cv2.putText(text_area, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # ترکیب تصویر اصلی با ناحیه متن
    result = np.vstack((image, text_area))
    
    return result

def transform(mx,my,H):
    
    point = np.array([mx, my], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point.reshape(1, -1, 2), np.linalg.inv(H))
    #transformed = cv2.perspectiveTransform(point.reshape(1, -1, 2), GT2)
    new_point = (transformed[0][0][0], transformed[0][0][1])
    center_point = (new_point[0], new_point[1])
    return center_point
    
def draw_circle(image,center_point):
  #center_point = center_point
  center_point= (int(center_point[0]), int(center_point[1]))
  # شعاع دایره
  radius = 5
  # رنگ دایره (BGR format)
  color = (255, 0, 0)  # سبز
  # ضخامت خط (اگر منفی باشد دایره پر میشود)
  thickness = -1
  # کشیدن دایره روی تصویر
  cv2.circle(image, center_point, radius, color, thickness)
  return image

def extend_line_to_image_borders_by_height(line_points, image_height=720, image_width=1280):
    """
    خط را تا مرزهای تصویر بر اساس ارتفاع گسترش می‌دهد
    
    Parameters:
    line_points: لیست شامل [x1, y1, x2, y2]
    image_height: ارتفاع تصویر
    image_width: عرض تصویر
    
    Returns:
    extended_line: نقاط خط گسترش یافته [x1, y1, x2, y2]
    slope: شیب خط
    equation: معادله خط
    """
    
    # استخراج نقاط
    x1, y1, x2, y2 = line_points
    
    # محاسبه شیب
    if x2 - x1 == 0:  # خط عمودی
        slope = float('inf')
        # برای خط عمودی، مختصات x ثابت می‌ماند
        extended_line = [x1, 0, x1, image_height]
        equation = f"x = {x1}"
        
    else:
        slope = (y2 - y1) / (x2 - x1)
        
        # محاسبه عرض از مبدأ (b)
        b = y1 - slope * x1
        
        # گسترش به بالا (y = 0)
        x_top = (0 - b) / slope if slope != 0 else x1
        x_top = int(round(x_top))
        
        # گسترش به پایین (y = image_height)
        x_bottom = (image_height - b) / slope if slope != 0 else x1
        x_bottom = int(round(x_bottom))
        
        # اطمینان از اینکه نقاط در محدوده تصویر هستند
        x_top = max(0, min(x_top, image_width))
        x_bottom = max(0, min(x_bottom, image_width))
        
        extended_line = [x_top, 0, x_bottom, image_height]
        equation = f"y = {slope:.4f}x + {b:.2f}"
    
    return extended_line, slope, equation

import numpy as np
import cv2

def extend_line_to_image_borders(line_points, image_width):

    # استخراج نقاط
    x1, y1, x2, y2 = line_points
    
    # محاسبه شیب
    if x2 - x1 == 0:  # خط عمودی
        slope = float('inf')
        # برای خط عمودی، مختصات x ثابت می‌ماند
        extended_line = [x1, 0, x1, image_width]
        equation = f"x = {x1}"
        
    else:
        slope = (y2 - y1) / (x2 - x1)
        
        # محاسبه عرض از مبدأ (b)
        b = y1 - slope * x1
        
        # گسترش به سمت چپ (x = 0)
        y_left = slope * 0 + b
        y_left = int(round(y_left))
        
        # گسترش به سمت راست (x = image_width)
        y_right = slope * image_width + b
        y_right = int(round(y_right))
        
        extended_line = [0, y_left, image_width, y_right]
        equation = f"y = {slope:.4f}x + {b:.2f}"
    
    return extended_line, slope, equation
    
