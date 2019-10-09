import numpy as np
import shutil
import glob

path = ["./1/*.JPG","./2/*.JPG","./3/*.JPG","./4/*.JPG"]


for i, p in enumerate(path):
    print(p)
    imgList = glob.glob(p)

    for img in imgList:
        img = img.replace("JPG","txt")
        fo = open(img,"w")
        fo.write(str(i))
        fo.close()


    


