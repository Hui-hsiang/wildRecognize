from keras.models import Sequential
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import glob
import numpy as np
import cv2

def centering_image(img):
    size = [256,256]
    
    img_size = img.shape[:2]
    print(img_size)
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized

def GetImg(image_path):
    image = cv2.imread(image_path)
    return image
def GetLabel(image_path):
    label_path = image_path.replace("JPG","txt")
    with open(label_path) as f:
        label = f.readlines()[0]
        label = int(label)
        #print(label)
    return label


def data_preprocessing(image):
    if(image.shape[0] > image.shape[1]):
            tile_size = (int(image.shape[1]*256/image.shape[0]),256)
    else:
            tile_size = (256, int(image.shape[0]*256/image.shape[1]))

    #centering
    image = centering_image(cv2.resize(image, dsize=tile_size))
        
    #out put 224*224px 
    image = image[16:240, 16:240]
    
    return image

images = []
labels = []

test_images_path = './test/*.jpg'
imagesList = glob.glob(test_images_path)

for img in imagesList:
    #print (img)
    image = GetImg(img)
    image = data_preprocessing(image)
    images.append(image)
    

images = np.array(images)
images = images.astype('float32')
images /= 255

# 載入訓練好的模型
net = load_model('yun_model.model')
predictions = net.predict(images)




for i, pre in enumerate(imagesList):
    pre = pre.replace("jpg","txt")
    with open(pre,'w') as fo:
        if predictions[i,0] > predictions[i,1]:
                fo.write("山羌")
        else:
                fo.write("沒有山羌")



