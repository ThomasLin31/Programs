# -*- coding: utf-8 -*-

import tkinter as tk
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

model1 =tf.keras.models.load_model('test_model.h5') # 之前訓練的模型


class Panel(object):


    def __init__(self):
        
        self.root = tk.Tk()
        self.root.title("簡易手寫辨識")
        self.model = model1 # 載入模型
        
        
        # 辨識按鈕
        self.recognize = tk.Button(self.root, text='辨識',width=8,height=3,font=("微軟正黑體", 10), command=self.Predict)
        self.recognize.grid(row=0, column=0, padx=5, pady=5)
        
        # 清除按鈕
        self.clear_button = tk.Button(self.root, text='清除',width=8,height=3,font=("微軟正黑體", 10), command=self.Clear)
        self.clear_button.grid(row=0, column=1, padx=5, pady=5)

        # 畫布
        self.canvas = tk.Canvas(self.root, bg='white', width=170, height=200)
        self.canvas.grid(row=1, columnspan=2, padx=5, pady=5)


        # 預測結果
        self.result = tk.Text(self.root, height=1, width=10,spacing1=30,spacing3=30,
                                        borderwidth=0,font=("微軟正黑體", 15), highlightthickness=0,
                                        relief='ridge')
        self.result.tag_configure("center", justify='center')
        self.result.grid(row=1, column=2)

        # 結果圖片
        self.image = tk.Canvas(self.root, width=140, height=150,highlightthickness=0, relief='ridge')
        self.image.create_image(0, 0, anchor="nw", tags="pltimg")
        self.image.grid(row=2, columnspan=2, padx=5, pady=5)
        
        # 鼠標事件
        self.last_x = -1
        self.last_y = -1
        self.canvas.bind('<B1-Motion>', self.Paint) # 左鍵按下後開始畫
        self.canvas.bind('<ButtonRelease-1>', self.StopPaint) # 鬆開則停止
        
        self.root.mainloop()
    
    # 顯示結果圖片
    def ShowImg(self, image):
        self.image.delete("pltimg")
        size = (150, 150)
        resized = image.resize(size, Image.ANTIALIAS)
        self.shownimg = ImageTk.PhotoImage(resized)
        
        self.image.create_image(0, 0, image=self.shownimg, anchor="nw", tags="pltimg")
    
    # 停止
    def StopPaint(self, event):
        self.last_x = -1
        self.last_y = -1
        
    # 清除
    def Clear(self):
        self.image.delete("pltimg")
        self.result.delete(1.0, "end") # 清空文字
        self.canvas.delete("all") # 清空畫布
        self.last_x = -1
        self.last_y = -1
        

    # 畫
    def Paint(self, event):

        if self.last_x !=-1 and self.last_y !=-1:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                               width=12, fill="black",
                               capstyle="round")
            
        self.last_x = event.x # 保留上次滑鼠位置
        self.last_y = event.y # 保留上次滑鼠位置

    # 預測
    def Predict(self):
        self.canvas.postscript(file="images/tmp.png") # 保存畫出的數字
        im = Image.open("images/tmp.png").convert('L') # 轉灰階
        width = float(im.size[0]) # 寬
        height = float(im.size[1]) # 高
        newImage = Image.new('L', (28, 28), (255))  # 新空白圖片
    
        # 待調整與優化，就是讓手寫的數字盡量符合訓練圖片的位置，以提高辨識度
        if width > height:  
            nheight = int(round((20.0 / width * height), 0)) 
            if (nheight == 0):
                nheight = 1

            img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN) # 銳利化
            wtop = int(round(((28 - nheight) / 2), 0))  
            newImage.paste(img, (4, wtop))  
          
        else:
            nwidth = int(round((20.0 / height * width), 0))  
            if (nwidth == 0):  
                nwidth = 1
               
            img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN) # 銳利化
            wleft = int(round(((28 - nwidth) / 2), 0))  
            newImage.paste(img, (wleft, 4))  
    
    
        tv = list(newImage.getdata())  # 圖片pixel值
    
        # 正規化 255 --> 1
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
        myimg=np.array(tva)
        myimg=myimg.reshape(28,28)
        plt.imshow(myimg.reshape(28, 28),cmap='Greys')
        savemyimg=(myimg.reshape(28, 28))
        plt.imsave("images/myimg.png",savemyimg)
        pltimg = Image.open("images/myimg.png").convert('L')
        self.ShowImg(pltimg)
        pred2 = model1.predict(myimg.reshape(1, 28, 28, 1)) # 預測
        print("預測的數字: ",pred2.argmax()) # 印出陣列中最大的數值
        self.result.insert("end", "預測為 {}".format(pred2.argmax()), 'center') # 顯示結果


if __name__ == '__main__':
    Panel()
