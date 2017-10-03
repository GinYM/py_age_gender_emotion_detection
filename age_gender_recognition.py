
# coding: utf-8

# ##Age and Gender Classification Using Convolutional Neural Networks - Demo
# 
# This code is released with the paper:
# 
# Gil Levi and Tal Hassner, "Age and Gender Classification Using Convolutional Neural Networks," IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015
# 
# If you find the code useful, please add suitable reference to the paper in your work.

# In[1]:

import cv2
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']

def age_gender_detection(frame,faces,age_net,gender_net):

	font = cv2.FONT_HERSHEY_SIMPLEX 
	age_prediction = []
	gender_prediction = []
	for x1,y1,x2,y2,score in faces:
		img = frame[int(y1):int(y2),int(x1):int(x2),:]
		prediction1 = age_net.predict([img])
		age_prediction.append(prediction1.argmax())
		prediction2 = gender_net.predict([img])
		gender_prediction.append(prediction2.argmax())
		cv2.putText(frame,'Age: '+age_list[prediction1.argmax()]+' Gender: '+gender_list[prediction2.argmax()],(int(x1),int(y1)-5), font, 0.5,(255,255,255),1)

	return frame

