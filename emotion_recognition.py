import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#import Image
import caffe
import scipy.io as sio


categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

plotSideBySide = True # Plot before/after images together?
saveDir = 'test_screenshots' # Folder to save screenshots to

useCNN = True # Set to false to simply display the default emoji
defaultEmoji = 2 # Index of default emoji (0-6)

### START SCRIPT ###

# Set up face detection

# Set up network
def make_net(mean=None):
    # net_dir specifies type of network 
    # Options are: (rgb, lbp, cyclic_lbp, cyclic_lbp_5, cyclic_lbp_10)

    caffe_root = '/media/gin/hacker/caffe-master'
    sys.path.insert(0, caffe_root + 'python')

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    net_root = 'model/emotion'

    net_pretrained = os.path.join(net_root, 'EmotiW_VGG_S.caffemodel')
    net_model_file = os.path.join(net_root, 'deploy.prototxt')
    VGG_S_Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
    return VGG_S_Net

def loadMeanCaffeImage(img="mean.binaryproto",curDir="model/emotion"):
  mean_filename=os.path.join(curDir,img)
  proto_data = open(mean_filename, "rb").read()
  a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
  mean  = caffe.io.blobproto_to_array(a)[0]
  return mean

def classify_video_frame(frame, faces, VGG_S_Net, categories=None):
    # Handle incorrect image dims for uncropped images
    # TODO: Get uncropped images to import correctly
    #if frame.shape[0] == 3:
    #    frame = np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2)


    # Convert to float format:
    frame = frame.astype(np.float32)
    frame /= 255.0

    labels = []

    for x,y,x2,y2,score in faces:
        img = frame[int(y):int(y2),int(x):int(x2),:]

        # Input image should be WxHxK, e.g. 490x640x3
        prediction = VGG_S_Net.predict([img], oversample=False)

        labels.append(prediction.argmax())

    return labels

def toggleRGB(img):
  r,g,b = cv.split(img)
  img = cv.merge([b,g,r])
  return img


def loadUintImage(imgFile):
  img = toggleRGB(caffe.io.load_image(imgFile))
  img *= 255.0
  img = img.astype(np.uint8)
  return img

def loadAllEmojis(emojiDir=None, categories=None):
    if emojiDir is None:
        emojiDir = 'dataset/'
    if categories is None:
        categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
  
    emojis = []
    for cat in categories:
        emojiFile = emojiDir + cat +'.mat'
        data = sio.loadmat(emojiFile)
        emojis.append(toggleRGB(data['img']))

    return emojis

def addEmojis(img,faces,emojis,labels):
    categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
    h,w,d = img.shape
    #print(h)
    #print(w)
    for i in range(len(labels)):
        x,y,x2,y2,score = faces[i]
        mid_y = int((y+y2)/2)
        mid_x = int((x+x2)/2)
        label = labels[i]
        emoji = emojis[int(label)]
        x_len = emoji.shape[0]
        y_len = emoji.shape[1]
        if x_len > min(x2,w)-x or y_len > min(y2,h)-y:
          continue
        y_start = mid_y-int(y_len/2)
        x_start = mid_x - int(x_len/2)
        #print(y_start)
        #print(x_start)

        # Resize emoji to desired width and height
        #dim = max(w,h)
        #em = cv.resize(emoji, (dim,dim), interpolation = cv.INTER_CUBIC)

        # Get boolean for transparency
        trans = emoji.copy()
        trans[emoji == 0] = 1
        trans[emoji != 0] = 0
        #trans[emoji == 255] = 1
        #trans[emoji == 0] = 0


        # Delete all pixels in image where emoji is nonzero
        
        if(y_start+y_len > h or x_start+x_len > w):
          continue

        img[y_start:y_start+y_len,x_start:x_start+x_len,:] *= trans

        # Add emoji on those pixels
        img[y_start:y_start+y_len,x_start:x_start+x_len,:] += emoji

    return img 