#-*- coding: utf-8 -*-

from MTCNN import draw_box
import cv2
import sys
from PIL import Image
import set_caffe
import caffe
from emotion_recognition import *
import sys
import numpy as np
from age_gender_recognition import *

caffe.set_mode_gpu()
caffe.set_device(0)
categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
emojis = loadAllEmojis()
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']


#for i in range(0,len(categories)):
#	#print(emojis[i])
#	print(emojis[i].shape)
#	np.savetxt(categories[i],emojis[i][:,:,1])

def getCamera(window_name,camera_index):
	cv2.namedWindow(window_name)
	cap = cv2.VideoCapture(camera_index)

	if cap.isOpened() == False:
		print("Open camera failed!\n")
	else:
		print("Open Camera sucessfuly![Press q to quit]\n")	

	count = 0
	PNet = caffe.Net("./model/face_detection/det1.prototxt", "./model/face_detection/det1.caffemodel", caffe.TEST)
	RNet = caffe.Net("./model/face_detection/det2.prototxt", "./model/face_detection/det2.caffemodel", caffe.TEST)
	ONet = caffe.Net("./model/face_detection/det3.prototxt", "./model/face_detection/det3.caffemodel", caffe.TEST)

	#emotion net
	mean = loadMeanCaffeImage()
	VGG_S_Net = make_net(mean)

	#age net
	mean_filename='./model/age_gender/mean.binaryproto'
	proto_data = open(mean_filename, "rb").read()
	a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
	mean  = caffe.io.blobproto_to_array(a)[0]
	age_net_pretrained='./model/age_gender/age_net.caffemodel'
	age_net_model_file='./model/age_gender/deploy_age.prototxt'
	age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
					mean=mean,
					channel_swap=(2,1,0),
					raw_scale=255,
					image_dims=(256, 256))

	#gender net
	gender_net_pretrained='./model/age_gender/gender_net.caffemodel'
	gender_net_model_file='./model/age_gender/deploy_gender.prototxt'
	gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
					mean=mean,
					channel_swap=(2,1,0),
					raw_scale=255,
					image_dims=(256, 256))


	while cap.isOpened():
		count = (count+1)%1000

		state,frame = cap.read()
		if state == False:
			print("Please check your camera!\n")
			break

		
		[frame,faces] = draw_box(frame,PNet,RNet,ONet)
		#print(faces)
		

		#emotion recognition
		cvImg = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
		labels = classify_video_frame(cvImg, faces, VGG_S_Net, categories=None)
		for idx in labels:
		 	sys.stdout.write(categories[idx])
		 	sys.stdout.flush()
		 	sys.stdout.write(' ')
		 	sys.stdout.flush()
		print('')

		
		frame = age_gender_detection(frame,faces,age_net,gender_net)
		frame_emotion = addEmojis(frame,faces,emojis,labels)
		cv2.imshow(window_name,frame_emotion)

		#for i in range(len(age_prediction)):
		#	print('Age: '+age_list[age_prediction[i]]+' Gender: '+gender_list[gender_prediction[i]])
		


		c = cv2.waitKey(10)
		if c & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: {} camera_id\n".format(sys.argv[0]))
	else:
		getCamera("Age Gender Emotion Detection",int(sys.argv[1]))