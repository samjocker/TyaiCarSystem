import colorsys
import copy
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image

#--------------------------------------------#
#   使用自己訓練好的模型預測需要修改3個參數
#   model_path、backbone和num_classes都需要修改！
#   如果出現shape不匹配
#   一定要注意訓練時的model_path、
#   backbone和num_classes數的修改
#--------------------------------------------#
class DeeplabV3(object):
    _defaults = {
        "model_path": 'model/3_7.h5',
        "num_classes": 7,
        "backbone": "mobilenet",
        "input_shape": [387, 688],
        "downsample_factor": 16,
        "mix_type": 0,
    }

    #---------------------------------------------------#
    #   初始化Deeplab
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   畫框設置不同的顏色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   獲得模型
        #---------------------------------------------------#
        self.generate()


    #---------------------------------------------------#
    #   獲得所有的分類
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   載入模型與權值
        #-------------------------------#
        self.model = Deeplabv3([self.input_shape[0], self.input_shape[1], 3], self.num_classes,
                                backbone = self.backbone, downsample_factor = self.downsample_factor)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))
        
    @tf.function
    def get_pred(self, image_data):
        pr = self.model(image_data, training=False)
        return pr
    #---------------------------------------------------#
    #   檢測圖片
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        #   在這裡將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
        #   代碼僅僅支持RGB圖像的預測，所有其它類型的圖像都會轉化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   對輸入圖像進行一個備份，後面用於繪圖
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別 
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   歸一化+通道數調整到第一維度+添加上batch_size維度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        #---------------------------------------------------#
        #   圖片傳入網絡進行預測
        #---------------------------------------------------#
        pr = self.get_pred(image_data)[0].numpy()
        #---------------------------------------------------#
        #   將灰條部分截取掉
        #---------------------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   進行圖片的resize
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   取出每一個像素點的種類
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        #   計數
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        self.mix_type = 0

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   將新圖片轉換成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   將新圖與原圖及進行混合
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   將新圖片轉換成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   將新圖片轉換成Image的形式
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        return image,pr

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   在這裡將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
        #   代碼僅僅支持RGB圖像的預測，所有其它類型的圖像都會轉化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別 
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   歸一化+通道數調整到第一維度+添加上batch_size維度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        #---------------------------------------------------#
        #   圖片傳入網絡進行預測
        #---------------------------------------------------#
        pr = self.get_pred(image_data)[0].numpy()
        #---------------------------------------------------#
        #   取出每一個像素點的種類
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])
        #--------------------------------------#
        #   將灰條部分截取掉
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
                
        t1 = time.time()
        for _ in range(test_interval):
            #---------------------------------------------------#
            #   圖片傳入網絡進行預測
            #---------------------------------------------------#
            pr = self.get_pred(image_data)[0].numpy()
            #---------------------------------------------------#
            #   取出每一個像素點的種類
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])
            #--------------------------------------#
            #   將灰條部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在這裡將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
        #   代碼僅僅支持RGB圖像的預測，所有其它類型的圖像都會轉化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別 
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   歸一化+通道數調整到第一維度+添加上batch_size維度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        #---------------------------------------------------#
        #   圖片傳入網絡進行預測
        #---------------------------------------------------#
        pr = self.get_pred(image_data)[0].numpy()
        #--------------------------------------#
        #   將灰條部分截取掉
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #--------------------------------------#
        #   進行圖片的resize
        #--------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   取出每一個像素點的種類
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
