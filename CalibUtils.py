import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Arc
import math


class CalibUtils:
    
    def __init__(self):
        
        self.lines_coords = [[[0., 54.16, 0.], [16.5, 54.16, 0.]],
                    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                    [[16.5, 13.84, 0.], [0., 13.84, 0.]],
                    [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
                    [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                    [[0., 37.66, -2.44], [0., 30.34, -2.44]],
                    [[0., 37.66, 0.], [0., 37.66, -2.44]],
                    [[0., 30.34, 0.], [0., 30.34, -2.44]],
                    [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                    [[105., 30.34, 0.], [105., 30.34, -2.44]],
                    [[105., 37.66, 0.], [105., 37.66, -2.44]],
                    [[52.5, 0., 0.], [52.5, 68, 0.]],
                    [[0., 68., 0.], [105., 68., 0.]],
                    [[0., 0., 0.], [0., 68., 0.]],
                    [[105., 0., 0.], [105., 68., 0.]],
                    [[0., 0., 0.], [105., 0., 0.]],
                    [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
                    [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                    [[99.5, 43.16, 0.], [105., 43.16, 0.]],
                    [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                    [[99.5, 24.84, 0.], [105., 24.84, 0.]]]
    
    def calculate_angle(self,line1, line2):
        """
        line1: list of 2 points (each point is a list or array of 3 elements)
        line2: same as line1
        """
        # استخراج نقاط (حذف کردن عنصر سوم)
        p1_start = np.array(line1[0][:2])
        p1_end = np.array(line1[1][:2])
        
        p2_start = np.array(line2[0][:2])
        p2_end = np.array(line2[1][:2])
        
        # محاسبه بردارهای دو خط
        v1 = p1_end - p1_start
        v2 = p2_end - p2_start
        
        # نرمال‌سازی بردارها
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # ضرب داخلی و محاسبه زاویه
        dot_product = np.dot(v1_norm, v2_norm)
        # محدود کردن برای جلوگیری از خطای دقت عددی
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
        
    def angle_difference(self,angle1, angle2):
        # اختلاف مطلق بین دو زاویه
        diff = abs(angle1 - angle2)
        # کوتاه‌ترین مسیر روی دایره (بین 0 و 180)
        return min(diff, 180 - diff)
        
    def get_unique_color(self,index, total_items):
       """Generates a unique BGR color based on the index."""
       # Using a colormap to get distinct colors
       cmap = plt.cm.get_cmap('hsv', total_items)
       rgb_color = cmap(index)[:3] # Get RGB values from colormap
       bgr_color = (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))
       return bgr_color
       
    def projection_from_cam_params(self,final_params_dict):
       cam_params = final_params_dict["cam_params"]
       x_focal_length = cam_params['x_focal_length']
       y_focal_length = cam_params['y_focal_length']
       principal_point = np.array(cam_params['principal_point'])
       position_meters = np.array(cam_params['position_meters'])
       rotation = np.array(cam_params['rotation_matrix'])
    
       It = np.eye(4)[:-1]
       It[:, -1] = -position_meters
       Q = np.array([[x_focal_length, 0, principal_point[0]],
                     [0, y_focal_length, principal_point[1]],
                     [0, 0, 1]])
       P = Q @ (rotation @ It)
    
       return P

    

    
    
    def line_direction(self,line_points):
        
        # تبدیل نقاط به مختصات دوبعدی (حذف مقدار همگن)
        p1 = line_points[0][:2]
        p2 = line_points[1][:2]
    
        # محاسبه بردار جهت
        direction = p2 - p1
    
        # نرمال‌سازی
        norm = np.linalg.norm(direction)
        unit_direction = direction / norm
    
        return direction, unit_direction
    '''
    70-105 tirack
    7.5 <
    
    70-125 markazi
    70-125
    7.5<
    
    7.5< DIFF
    '''
    def check_points(self,points, w, h):
        result = []
        neg=0;pos=0;
        for point in points:
            x, y, _ = point  # نقطه در بعد x و y و یک مقدار اضافی
            if not (0 < x <= w):  # چک کردن محدوده x
                x = -1
                neg+=1;
            else:
                x =  2;
                pos+=1;
                
            if not (0 < y <= h):  # چک کردن محدوده y
                y = -1
                neg+=1;
            else:
                y = 2;
                pos+=1;
            result.append([x, y])  # 
            
        return result,neg,pos
        
    def logit2(self,lino,h,w):
    
        R = [3,4,5,20,21,22,10,9,11,15]
        L = [0,2,1,17,18,19,6,7,8,14]
    
        T6 = lino[6]
        T7 = lino[7]
        T8 = lino[8]
    
        T9 = lino[9]
        T10 = lino[10]
        T11 = lino[11]
    
        T12 = lino[12]
        T16 = lino[16]
        T13 = lino[13]
    
        T5 = lino[5]
        T3 = lino[3]
        T20 = lino[20]
        T22 = lino[22]
    
        T21 = lino[21]
        T4 = lino[4]
        T15 = lino[15]
    
        T18 = lino[18]
        T14 = lino[14]
        T1 = lino[1]
    
    
        T17 = lino[17]
        T19 = lino[19]
        T0 = lino[0]
        T2 = lino[2]
    
        import numpy as np
    
        
        sR=0;negTr=0;posTr=0;
        for r in R:
            #print('ex r ',r,lino[r])
            linoc,neg,pos = self.check_points(lino[r], w, h)
            negTr+=neg;posTr+=pos; 
            #print('af r ',r,linoc)
            
            #sR+=np.sign(lino[r])
    
        sL=0;negTl=0;posTl=0;
        for l in L:
            #print('ex l ',l,lino[l])
            linoc,neg,pos = self.check_points(lino[l], w, h)
            negTl+=neg;posTl+=pos; 
            #print('af l ',r,linoc)
            
            #sL+=np.sign(lino[l])
    
        #print('-----------------------------------')
        
        #print('negTr',negTr,'posTr',posTr,)
        #print('negTl',negTl,'posTl',posTl,)
    
        Rdiff = posTr-negTr
        Ldiff = posTl-negTl
    
        #print('Rdiff',Rdiff,'Ldiff',Ldiff,)
    
        info={}
        info.update({'Rdiff':Rdiff,'Ldiff':Ldiff})
    
        dirx=''
        if min(info['Rdiff'],info['Ldiff'])<0:
            if info['Rdiff']>info['Ldiff'] and (info['Rdiff']-info['Ldiff'])>=6:
                dirx='R'
            if info['Rdiff']<info['Ldiff'] and (info['Ldiff']-info['Rdiff'])>=6:
                dirx='L'
                
        
        
    
        info.update({'dir':dirx})
    
        
    
        ############ TIRACK ######################
        A67 = self.calculate_angle(T6, T7)
        A68 = self.calculate_angle(T6, T8)
        A78 = self.calculate_angle(T7, T8)
    
        A0910 = self.calculate_angle(T9, T10)
        A0911 = self.calculate_angle(T9, T11)
        A1011 = self.calculate_angle(T10, T11)
    
        info.update({'A67':A67,'A68':A68,'A78':A78,'A0910':A0910,'A0911':A0911,'A1011':A1011})
        
        t1=70; t2=105; t3=7.5
        
        Flag=1;
    
        if dirx=='L':
            pass
        else:
            t2 = t2 + 70
            t3 = t3 + 30
        
        if A67>=t1 and A67<=t2 and A68>=t1 and A68<=t2 and A78<=t3:
            pass
        else:
            Flag=0;
    
    
        #print('Flag 1',Flag)
    
    
        t1b=70; t2b=105; t3b=7.5
    
        if dirx=='R':
            pass
        else:
            t2b = t2b + 70
            t3b = t3b + 30
            
        if A0910>=t1b and A0910<=t2b and A0911>=t1b and A0911<=t2b and A1011<=t3b:
            #print('ok 2',info)
            pass
        else:
            Flag=0;
            
    
        #print('Flag 2',Flag)
    
        
        
        ############ Center ##################
        t4=65; t5=125; t6=7.5
    
        if dirx!='':
            t4=t4-25
            t5=t5+25
            t6=t6+10
            
        A1213 = self.calculate_angle(T12, T13)
        A1216 = self.calculate_angle(T12, T16)
        A1316 = self.calculate_angle(T13, T16)
    
        A1316 = min(180-A1316,A1316)
        info.update({'A1213':A1213,'A1216':A1216,'A1316':A1316})
    
        if A1213>=t4 and A1213<=t5 and A1216>=t4 and A1216<=t5 and A1316<=t6:
    
            #print('ok 3',info)
            pass
        else:
            #print('n ok 3',info)
            Flag=0;
    
        #print('Flag 3',Flag)
    
    
        ####### DIFF ################
        
        t7=7.5
        if dirx!='':
            t7=t7+5
            
        angle1 = self.calculate_angle(T5, T3)
        angle2 = self.calculate_angle(T20, T22)
    
        angle3 = self.calculate_angle(T21, T15)
        angle4 = self.calculate_angle(T21, T4)
    
        angle5 = self.calculate_angle(T18, T14)
        angle6 = self.calculate_angle(T18, T1)
    
        angle7 = self.calculate_angle(T17, T19)
        angle8 = self.calculate_angle(T0, T2)
    
        Diff1 = self.angle_difference(angle1, angle2);
        Diff2 = self.angle_difference(angle3, angle4)
        Diff3 = self.angle_difference(angle5, angle6)
        Diff4 = self.angle_difference(angle7, angle8)
    
        info.update({'Diff1':Diff1,'Diff2':Diff2,'Diff3':Diff3,'Diff4':Diff4})
    
    
        if Diff1<=t7 and Diff2<=t7 and Diff3<=t7 and Diff4<=t7 and np.mean([Diff1,Diff2,Diff3,Diff4])>=0.5 :
    
            #print('ok 4',info)
            
            pass
        else:
            Flag=0;
    
    
        #print('Flag 4',Flag)
    
        
        ############ SHAK #########################
        T0 = lino[0]
        T1 = lino[1]
    
        J15 = np.sum(np.abs(self.line_direction(T15)[0]))
        J14 = np.sum(np.abs(self.line_direction(T14)[0]))
    
        info.update({'J15':J15,'J14':J14})
    
        if max(J14,J15)>=1200:
    
            #print('ok 5',info)
            
            pass
        else:
            Flag=0;
    
    
        #print('Flag 5',Flag)
    
        
        return Flag,info
    
    

    
    def warp_point_homogeneous(self,matrix, point_homogeneous):
        # point_homogeneous: np.array([x, y, 1])
        transformed = matrix @ point_homogeneous
        transformed /= transformed[2]  # Normalize: divide by w
        return np.array([transformed[0], transformed[1], 1.0])  # حفظ 1 در آخر
    
    def undox(self,lino,matrix2):
        warped_pairs = []
        for pair in lino:
            warped_pair = []
            for point in pair:
                warped = self.warp_point_homogeneous(matrix2, point)
                warped_pair.append(warped)
            warped_pairs.append(warped_pair)
        return warped_pairs


    
    
    def project(self,frame, P,indexq,matrix_inv):
    
        lino=[]
        for i, line in enumerate(self.lines_coords):
            color = self.get_unique_color(i, len(self.lines_coords))
        
            w1 = line[0]
            w2 = line[1]
            i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
            i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
            i1 /= i1[-1]
            i2 /= i2[-1]
            #frame = cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), color, 3)
        
            mid_x = int((i1[0] + i2[0]) / 2)
            mid_y = int((i1[1] + i2[1]) / 2)
            #cv2.putText(frame, f'{i}', (mid_x + 5, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            lino.append([i1,i2])
    
        #lino = self.rotate_point_list(lino,theta_deg=-30)
        
        if matrix_inv is not None:
            lino = self.undox(lino,matrix_inv)


        w=frame.shape[0]
        h=frame.shape[1]
        Flag,info = self.logit2(lino,w,h)
    
    
        return frame,info,Flag,lino
    
    def draw_pitch(self,ax):
        
        # ابعاد زمین
        pitch_length = 105
        pitch_width = 68
    
        cords = {}
        label = 1
    
        # گوشه‌های زمین
        corners = [
            (0,0), (pitch_length,0), (pitch_length,pitch_width), (0,pitch_width)
        ]
        for point in corners:
            cords[label] = point
            ax.text(point[0], point[1], f'{label}', color='r', fontsize=12)
            label +=1
    
        # گوشه‌های محوطه جریمه
        penalty_left = [(16.5,13.84), (0,13.84), (0,54.16), (16.5,54.16)]
        for point in penalty_left:
            cords[label] = point
            ax.text(point[0], point[1], f'{label}', color='b', fontsize=12)
            label +=1
    
        penalty_right = [(pitch_length-16.5,13.84), (pitch_length,13.84),
                         (pitch_length,54.16), (pitch_length-16.5,54.16)]
        for point in penalty_right:
            cords[label] = point
            ax.text(point[0], point[1], f'{label}', color='b', fontsize=12)
            label +=1
    
        # گوشه‌های شش‌قدم
        six_left = [(5.5,24.84), (0,24.84), (0,43.16), (5.5,43.16)]
        for point in six_left:
            cords[label] = point
            ax.text(point[0], point[1], f'{label}', color='g', fontsize=12)
            label +=1
    
        six_right = [(pitch_length-5.5,24.84), (pitch_length,24.84),
                     (pitch_length,43.16), (pitch_length-5.5,43.16)]
        for point in six_right:
            cords[label] = point
            ax.text(point[0], point[1], f'{label}', color='g', fontsize=12)
            label +=1
    
        # نقطه وسط زمین
        center_point = (pitch_length/2, pitch_width/2)
        cords[label] = center_point
        ax.text(center_point[0], center_point[1], f'{label}', color='m', fontsize=12)
        label +=1
    
        # برخورد خط وسط زمین با خطوط طولی
        top_mid = (pitch_length/2, pitch_width)
        bottom_mid = (pitch_length/2, 0)
    
        cords[label] = top_mid
        ax.text(top_mid[0], top_mid[1], f'{label}', color='c', fontsize=12)
        label +=1
    
        cords[label] = bottom_mid
        ax.text(bottom_mid[0], bottom_mid[1], f'{label}', color='c', fontsize=12)
        label +=1
    
        # رسم زمین
        ax.plot([0, pitch_length, pitch_length, 0, 0], [0,0,pitch_width,pitch_width,0], 'k')
        ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], 'k')
        ax.add_patch(plt.Circle(center_point, 9.15, color='k', fill=False))
        ax.plot(center_point[0], center_point[1], 'ko')
    
        ax.plot([0,16.5,16.5,0,0], [13.84,13.84,54.16,54.16,13.84], 'k')
        ax.plot([pitch_length, pitch_length-16.5, pitch_length-16.5, pitch_length, pitch_length],
                [13.84,13.84,54.16,54.16,13.84], 'k')
    
        ax.plot([0,5.5,5.5,0,0], [24.84,24.84,43.16,43.16,24.84], 'k')
        ax.plot([pitch_length, pitch_length-5.5, pitch_length-5.5, pitch_length, pitch_length],
                [24.84,24.84,43.16,43.16,24.84], 'k')
    
        ax.plot(11, pitch_width/2, 'ko')
        ax.plot(pitch_length-11, pitch_width/2, 'ko')
    
        left_arc = Arc((11, pitch_width/2), width=2*9.15, height=2*9.15, angle=0, theta1=308, theta2=52, color='k')
        right_arc = Arc((pitch_length-11, pitch_width/2), width=2*9.15, height=2*9.15, angle=0, theta1=128, theta2=232, color='k')
        ax.add_patch(left_arc)
        ax.add_patch(right_arc)
    
        ax.set_xlim(-5, pitch_length+5)
        ax.set_ylim(-5, pitch_width+5)
        ax.set_aspect('equal')
        ax.set_xlabel('L')
        ax.set_ylabel('W')
        #ax.set_title('زمین فوتبال استاندارد')
    
        return cords
    
    def line_vis(self,frame,P,info,Flag,lino,matrix_inv=None):
        for i, line in enumerate(self.lines_coords):
        
            i1,i2 = lino[i]
            
            if Flag==1:
                color = self.get_unique_color(i, len(self.lines_coords))
                frame = cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), color, 3)
                
            else:
                #color = (30,30,30)
                color = self.get_unique_color(i, len(self.lines_coords))
                overlay = frame.copy()
                cv2.line(overlay, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), color, thickness=3)
                alpha = 0.2
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
            
            mid_x = int((i1[0] + i2[0]) / 2)
            mid_y = int((i1[1] + i2[1]) / 2)
            cv2.putText(frame, f'{i}', (mid_x + 5, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
                
        r = 9.15
        pts1, pts2, pts3 = [], [], []
        base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
        for ang in np.linspace(37, 143, 50):
            ang = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
            ipos = P @ pos
            ipos /= ipos[-1]
            pts1.append([ipos[0], ipos[1]])
        
        base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
        for ang in np.linspace(217, 323, 200):
            ang = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
            ipos = P @ pos
            ipos /= ipos[-1]
            pts2.append([ipos[0], ipos[1]])
        
        base_pos = np.array([0, 0, 0., 0.])
        for ang in np.linspace(0, 360, 500):
            ang = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
            ipos = P @ pos
            ipos /= ipos[-1]
            pts3.append([ipos[0], ipos[1]])
        
        #print(pts1)
        if matrix_inv is not None:
            pts1 = self.warp_points_2d(pts1, matrix_inv)
            pts2 = self.warp_points_2d(pts2, matrix_inv)
            pts3 = self.warp_points_2d(pts3, matrix_inv)
         
        #print(pts1)

        XEllipse1 = np.array(pts1, np.int32)
        XEllipse2 = np.array(pts2, np.int32)
        XEllipse3 = np.array(pts3, np.int32)
        
        if Flag==1:
            frame = cv2.polylines(frame, [XEllipse1], False, color, 3)
            frame = cv2.polylines(frame, [XEllipse2], False, color, 3)
            frame = cv2.polylines(frame, [XEllipse3], False, color, 3)
            
        else:
            overlay = frame.copy()
            cv2.polylines(overlay, [XEllipse1], False, color, 3)
            cv2.polylines(overlay, [XEllipse2], False, color, 3)
            cv2.polylines(overlay, [XEllipse3], False, color, 3)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
        return frame
    
    def warp_points_2d(self,points, matrix):

        warped_points = []
        for pt in points:
            x, y = pt
            pt_hom = np.array([x, y, 1.0])
            transformed = matrix @ pt_hom
            transformed /= transformed[2]  # تقسیم بر w
            warped_points.append([transformed[0], transformed[1]])
        return warped_points



    def homog(self,mapp,cords):
        image_points=[];
        map_points = [];
        for key in mapp.keys():
            map_points.append(cords[key])
            image_points.append(mapp[key][:2])
            
        map_points = np.asarray(map_points)
        image_points = np.asarray(image_points)
    
        H, status = cv2.findHomography(image_points, map_points, cv2.RANSAC, 5.0)
        return H,status

    def apply_homog(self,H,detect,show=True):
    
        fig, ax = plt.subplots(figsize=(12,7))
        cords = self.draw_pitch(ax)
    
        for det in detect.xyxy:
            x1,y1,x2,y2 = det
    
            special_point = [x2, y2]
            
            point_h = np.array([special_point[0], special_point[1], 1.0]).reshape(3, 1)
            mapped_point_h = H @ point_h
            mapped_point = mapped_point_h[:2] / mapped_point_h[2]
            x_map, y_map = mapped_point.flatten()
            

            ax.scatter(x_map, y_map, color='orange', s=150, edgecolors='black', zorder=5)
    
    
        plt.savefig('temp.png')
    
        if show==True:
            plt.show()
    
        return cv2.imread('temp.png')
        

    def smooth_params(self,params_list, window_size=5):
        smoothed = []
        keys = params_list[0]["cam_params"].keys()
        half_window = window_size // 2
    
        for i in range(len(params_list)):
            start = max(0, i - half_window)
            end = min(len(params_list), i + half_window + 1)
            window = params_list[start:end]
            median_vals = {}
            for k in keys:
                values = np.array([p["cam_params"][k] for p in window])
                # اگر مقدار هر پارامتر برداری است (مثلاً آرایه یا ماتریس)، median رو به صورت عنصر به عنصر بگیر
                if values.ndim > 1:
                    median_val = np.median(values, axis=0)
                else:
                    median_val = np.median(values)
                median_vals[k] = median_val
            smoothed.append({"cam_params": median_vals})
        return smoothed

    
        
    def showinfo(self,frame,info,indexq):
        

        # Draw text at the bottom of the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)  # White color
        
        
            
        # Get frame dimensions
        h, w, _ = frame.shape
        goalpost = '{Frame' + str(indexq) + '}  '+ 'A67 '+str(info['A67'])[:4]+' | A68 '+str(info['A68'])[:4]+' | A78 '+str(info['A78'])[:4]+' | A0910 '+str(info['A0910'])[:4]+' | A0911 '+str(info['A0911'])[:4]+' | A1011 '+str(info['A1011'])[:4]
        center =   'A1213 '+str(info['A1213'])[:4]+' | A1216 '+str(info['A1216'])[:4]+' | A1316 '+str(info['A1316'])[:4]
        diff =  'Diff1 '+str(info['Diff1'])[:4]+' | Diff2 '+str(info['Diff2'])[:4]+' | Diff3 '+str(info['Diff3'])[:4] +' | Diff4 '+str(info['Diff4'])[:4] +' | J14 '+str(info['J14'])[:7] +' | J15 '+str(info['J15'])[:7] + ' | R '+str(info['Rdiff'])[:7] +' | L '+str(info['Ldiff'])[:7] + '| Di ' + str(info['dir'])
        
        
        #info.update({'Rdiff':Rdiff,'Ldiff':Ldiff})
        
        
        # تنظیمات مستطیل بکگراند
        bg_color = (0, 0, 0)  # مشکی
        alpha = 0.5  # شفافیت مستطیل، اگر نیاز به شفافیت داشتید (بین 0 و 1)
        
        # موقعیت متن و ابعاد
        (text_w1, text_h1), _ = cv2.getTextSize(goalpost, font, font_scale, font_thickness)
        (text_w2, text_h2), _ = cv2.getTextSize(center, font, font_scale, font_thickness)
        (text_w3, text_h3), _ = cv2.getTextSize(diff, font, font_scale, font_thickness)
        
        
        # محل رسم بکگراند
        cv2.rectangle(frame, (5, h - 100 - text_h1 - 10), (5 + text_w1 + 10, h - 60 + 10), bg_color, -1)
        cv2.rectangle(frame, (5, h - 60 - text_h2 - 10), (5 + text_w2 + 10, h - 20 + 10), bg_color, -1)
        cv2.rectangle(frame, (5, h - 20 - text_h3 - 10), (5 + text_w3 + 10, h - 20 + 10), bg_color, -1)
        
        # رسم متن‌ها
        cv2.putText(frame, goalpost, (10, h - 100), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, center, (10, h - 60), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, diff, (10, h - 20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return frame
    
    def create_video(self,result,frames,save_path,frame_width,frame_height,fps,showinfo=False):
        
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        for key in result.keys():
            P,info,Flag,lino,matrix_inv = result[key]
            frame = self.line_vis(frames[key],P,info,Flag,lino,matrix_inv)
            if showinfo==True:
                frame=self.showinfo(frame,info,key)
            out.write(frame)
           
        out.release()
          






    def rotate_image_3d(self,image, axis='x', angle_degrees=30):
        h, w = image.shape[:2]
        angle = math.radians(angle_degrees)
    
        # مرکز تصویر
        cx, cy = w / 2, h / 2
    
        # فاصله مجازی از دوربین
        d = max(w, h)
    
        # نقاط اصلی تصویر
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
    
        # محاسبه نقاط مقصد بسته به محور
        if axis.lower() == 'x':
            dy = h / 2 * math.sin(angle)
            dst_pts = np.float32([
                [0,      dy],
                [w,      dy],
                [w,  h - dy],
                [0,  h - dy]
            ])
        elif axis.lower() == 'y':
            dx = w / 2 * math.sin(angle)
            dst_pts = np.float32([
                [dx,     0],
                [w - dx, 0],
                [w - dx, h],
                [dx,     h]
            ])
        elif axis.lower() == 'z':
            # چرخش دوبعدی معمولی حول مرکز
            M = cv2.getRotationMatrix2D((cx, cy), angle_degrees, 1.0)
            return cv2.warpAffine(image, M, (w, h))
        else:
            raise ValueError("محور باید 'x'، 'y' یا 'z' باشد")
    
        # ساخت ماتریس پرسپکتیو و اعمال آن
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        matrix_inv = cv2.getPerspectiveTransform(dst_pts,src_pts)
        return cv2.warpPerspective(image, matrix, (w, h)),matrix_inv



    def enhanc1(self,image: np.ndarray, margin: int = 50, saturation_boost: float = 1.5):
    
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
    
        hsv_boosted = hsv.copy()
        hsv_boosted[..., 1] = np.where(
            mask > 0,
            np.clip(hsv[..., 1] * saturation_boost, 0, 255),
            hsv[..., 1],
        )
        image_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
    
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("هیچ ناحیه سبزی پیدا نشد.")
            return image
    
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, image.shape[1])
        y2 = min(y + h + margin, image.shape[0])
    
        # Start with a white canvas and paste the boosted area
        result = np.full_like(image_boosted, fill_value=0)
        result[y1:y2, x1:x2] = image_boosted[y1:y2, x1:x2]

        result,matrix_inv = self.rotate_image_3d(result,'x',-30)
        
        return result,matrix_inv

