import numpy as np
import importlib
import similar
import similar
importlib.reload(similar)
import pickle
import view
import glob
import cv2
import os
import re
from tqdm import tqdm

from lightglue.utils import load_image, rbd


class Calibration:
    def __init__(self,match_path):
        self.soccer_coords_ref = {1: (0.00, 0.00),2: (105.00, 0.00),3: (105.00, 68.00),4: (0.00, 68.00),5: (16.50, 13.84),6: (0.0, 13.84),7: (0.00, 54.16),8: (16.50, 54.16),9: (88.50, 13.84),10: (105.00, 13.84),11: (105.00, 54.16),12: (88.50, 54.16),13: (5.50, 24.84),14: (0.00, 24.84),15: (0.00, 43.16),16: (5.50, 43.16),17: (99.50, 24.84),18: (105.00, 24.84),19: (105.00, 43.16),20: (99.50, 43.16),21: (52.50, 34.00),22: (52.50, 68.00),23: (52.50, 0.00),24: (52.50, 43.15),25: (52.50, 24.85),26: (43.35, 34.00),27: (61.65, 34.00),28: (16.63, 26.79),29: (16.63, 41.21),30: (88.37, 41.21),31: (88.37, 26.79),32: (11.00, 34.00),33: (94.00, 34.00),34:(0.00, 30.34) , 35 : (0.00, 37.66), 36:(105.00, 30.34), 37:(105.00, 37.66)}
        
        soccer_coords_array=[]
        for key in self.soccer_coords_ref.keys():
            soccer_coords_array.append(self.soccer_coords_ref[key])
        self.soccer_coords_array = np.asarray(soccer_coords_array)

        self.MATCH_PATH = match_path

    def validation_operator_file(self):
        TP=0;FP=0;
        self.operator_calibration_file_validate={}
        for key in self.operator_calibration_file.keys():
            MAPPoints = self.operator_calibration_file[key]['points']
            image_pixels = []
            soccer_coords = []
            for item in MAPPoints:
                image_pixels.append([item['imagePixel']['x'], item['imagePixel']['y']])
                soccer_coords.append([item['soccerPoint']['x'], item['soccerPoint']['y']])

            if len(np.asarray(image_pixels)) >= 4:
                transformer_image2radar = view.ViewTransformer(source=np.asarray(image_pixels), target=np.asarray(soccer_coords))

                transformer_radar2image = view.ViewTransformer(
                    source=np.asarray(soccer_coords),
                    target=np.asarray(image_pixels)
                )
            
                frame_all_points = transformer_radar2image.transform_points(points=self.soccer_coords_array)
                total_valid_points = len(np.unique(frame_all_points))

                if total_valid_points>=8:
                    self.operator_calibration_file_validate.update({key:{'transformer_image2radar':transformer_image2radar,'transformer_radar2image':transformer_radar2image,'points':MAPPoints}})
                    TP+=1;
                else:
                    FP+=1;
                    #pass
                    #print('not valid --> ', key,total_valid_points)
                    #print('-----------------')

        print('TP '+str(TP) + ' FP '+str(FP) + ' | R '+str(TP*100/(TP+FP))[:4]+'%' )

    def load_file_operator(self,path):

        if os.path.exists(path):
            with open(path,'rb') as f:
                self.operator_calibration_file = pickle.load(f)
                self.validation_operator_file()
                self.load_ref_image(windows=0)
                self.gathar_all_query_image(windows=0)

                return self.refsImage,self.query_image_path,self.operator_calibration_file_validate,self.operator_calibration_file

        else:
            raise("operator_calibration_file not exists")


    def load_ref_image(self,windows=1):

        self.refsImage={}
        for key in self.operator_calibration_file_validate.keys():
            path = self.MATCH_PATH+key
            if windows==1:
                path = re.sub(r'/part_\d+/', '/', path)
            #path = path.replace('part_0002','part_0001').replace('part_0003','part_0001')
            if os.path.exists(path):
                self.refsImage.update({key:cv2.imread(path)})
            #else:
                #print(path)

    def gathar_all_query_image(self,windows=1):
        self.query_image_path=[]
        if windows==1:
            paths = glob.glob(self.MATCH_PATH+'epi*/*.jpg')
        else:
            paths = glob.glob(self.MATCH_PATH+'epi*/part*/*.jpg')
            
        for path in np.sort(paths):
            path = path.replace('\\','/')
            self.query_image_path.append(path)
        

    def propagete_calibration(self,method="gray",limit=None):

        if method=="gray":
            self.pair_socre = similar.base_gray_color(self.query_image_path,self.operator_calibration_file_validate,self.MATCH_PATH,self.refsImage)

        elif method=="tm":
            self.pair_socre = similar.base_tm(self.query_image_path,self.operator_calibration_file_validate,self.MATCH_PATH,self.refsImage,limit)

        elif method=="ssim":
            self.pair_socre = similar.base_ssim(self.query_image_path,self.operator_calibration_file_validate,self.MATCH_PATH,self.refsImage)

        elif method=="sift":
            self.pair_socre = similar.base_sift(self.query_image_path,self.operator_calibration_file_validate,self.MATCH_PATH,self.refsImage)

        elif method=="some":
            self.pair_socre = similar.base_some_color(self.query_image_path,self.operator_calibration_file_validate,self.MATCH_PATH,self.refsImage)
        

        return self.pair_socre

    def direct_refkeypoint(self,outpath,device,extractor,matcher,perfix="/root/console/"):
        

        allref_candid_path = []

        for full_folder_path in np.sort(glob.glob(self.MATCH_PATH+"epi*/part*")):
        
            allpathsort = np.sort(glob.glob(full_folder_path+'/*.jpg'))
    
            if 'episode_Z' not in full_folder_path:
                if len(allpathsort)>=3:
                    A=allpathsort[0]
                    C=allpathsort[-1]
                    if len(allpathsort)%2==0:
                        B=int(np.ceil((len(allpathsort))/2));B=allpathsort[B]
                    else:
                        B=int(np.ceil((len(allpathsort)-1)/2));B=allpathsort[B]
                    allpathsort=[];
                    allpathsort.append(A);allpathsort.append(B);allpathsort.append(C);
                    allpathsort=np.asarray(allpathsort)

                #print('AAAA',allpathsort)
                for path in allpathsort:
                    key = path.replace(self.MATCH_PATH,"")
                    allref_candid_path.append(perfix+path)

                    pathimage = perfix+path
                    savefile = pathimage.replace('.jpg','.pkl').split('/')
                    savefile = outpath+savefile[-3]+'_'+savefile[-2]+'_'+savefile[-1]

                    #print(savefile,os.path.exists(savefile))
                    #print(asd)
                    
                    if os.path.exists(savefile)==False:
                        query = load_image(pathimage)
                        similar.calculate_keypoint(pathimage,query,device,extractor,matcher,outpath)

                    #else:
                    #    print(pathimage," exsit!")
        
        allresult={}
        for path1 in tqdm(allref_candid_path):
            for path2 in allref_candid_path:


                key1 = path1.replace(self.MATCH_PATH,"").replace(perfix,"")
                key2 = path2.replace(self.MATCH_PATH,"").replace(perfix,"")

                pairkey1 = (key1,key2)
                pairkey2 = (key2,key1)
                
                if path1!=path2 and pairkey1 not in allresult.keys() and pairkey2 not in allresult.keys():
                    
                    image1 = load_image(path1)
                    image2 = load_image(path2)

                    pathq = path1.split("/"); pathq = pathq[-3]+"_"+pathq[-2]+"_"+pathq[-1]
                    pathr = path2.split("/"); pathr = pathr[-3]+"_"+pathr[-2]+"_"+pathr[-1]            

                    pathq = (outpath+pathq+'.pkl').replace('.jpg','')
                    pathr = (outpath+pathr+'.pkl').replace('.jpg','')

                    if os.path.exists(pathq):
                        with open(pathq,'rb') as f:
                            feats0 = pickle.load(f)
                            
                        if os.path.exists(pathr):
                            with open(pathr,'rb') as f:
                                feats1 = pickle.load(f)

                                #print(path1,path2)
                                #print(pathq,pathr)

                                #print(asd)
                                try:
                                    H_PAIR,H_PAIR_INV = similar.calculate_homography_between_frames_withkey(image1,image2,device,extractor,matcher,feats0,feats1)
                                    #H,H_INV = similar.calculate_homography_between_frames_novel(image1,image2,device,extractor,matcher)
                                    allresult.update({pairkey1:H_PAIR})
                                    allresult.update({pairkey2:H_PAIR_INV})

                                except:
                                    pass
                    #break
            #break
            
        savefile =  outpath +  "allref_candid_H.pkl"  
        with open(savefile,'wb') as f:
            pickle.dump(allresult,f)
        

    def direct_keypoint(self,outpath,device,extractor,matcher):
        self.query_image_path = []
        paths = glob.glob("/root/console/"+self.MATCH_PATH+'epi*/part*/*.jpg')
        for path in tqdm(np.sort(paths)):
            path = path.replace('\\','/')
            #self.query_image_path.append(path)

            savefile = path.replace('.jpg','.pkl').split('/')
            savefile = outpath+savefile[-3]+'_'+savefile[-2]+'_'+savefile[-1]

            if os.path.exists(savefile)==False:
                query = load_image(path)
                similar.calculate_keypoint(path,query,device,extractor,matcher,outpath)
        
            
    def direct_solution(self,CONFIG,method="tm",limit=None,perfix="/kaggle/working/",resizef=0.5):
        self.query_image_path = []
        paths = glob.glob(perfix+self.MATCH_PATH+'epi*/part*/*.jpg')
        for path in np.sort(paths):
            path = path.replace('\\','/')
            self.query_image_path.append(path)
            
        
        self.refsImage = {}

        for full_folder_path in np.sort(glob.glob(self.MATCH_PATH+"epi*/part*")):
        
            allpathsort = np.sort(glob.glob(full_folder_path+'/*.jpg'))
    
            if 'episode_Z' not in full_folder_path:
                if len(allpathsort)>=3:
                    A=allpathsort[0]
                    C=allpathsort[-1]
                    if len(allpathsort)%2==0:
                        B=int(np.ceil((len(allpathsort))/2));B=allpathsort[B]
                    else:
                        B=int(np.ceil((len(allpathsort)-1)/2));B=allpathsort[B]
                    allpathsort=[];
                    allpathsort.append(A);allpathsort.append(B);allpathsort.append(C);
                    allpathsort=np.asarray(allpathsort)

                #print('AAAA',allpathsort)
                for path in allpathsort:
                    key = path.replace(self.MATCH_PATH,"")
                    self.refsImage.update({key:cv2.imread(path)})

                    
        if method=="tm":
            #print('self.query_image_path',self.query_image_path)
            self.pair_socre = similar.base_tm(self.query_image_path,None,self.MATCH_PATH,self.refsImage,limit,resizef=resizef)
            #self.pair_socre = similar.base_tm_gpu(self.query_image_path, self.refsImage, resize_factor=0.5, limit=limit, device='cuda')
            return self.pair_socre

    def draw_circle(self,image,center_point):
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

    def draw_pitch_on_image(self,image,H):

        for key in self.soccer_coords_ref.keys():
            points = np.asarray(self.soccer_coords_ref[key])
            map1 = view.rawTransofrm(points,H)

            image = self.draw_circle(image,(map1[0,0],map1[0,1]))

        return image

    def add_text_below_image(self,image, text, background_color=(0, 0, 0), text_color=(255, 255, 255)):
        
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
        





