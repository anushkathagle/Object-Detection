#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import os


# In[10]:


LabelPath = "coco.names"
weightPath = "yolov3.weights"
confPath = "yolov3.cfg"

Labels = open(LabelPath).read().strip().split("\n")
YoloShape = (416, 416)
image = cv2.imread("Cyclists.jpg")
H, W = image.shape[:2]  #### we will use in for rectangle
Pre_image = cv2.dnn.blobFromImage(image, 1/255.0, YoloShape, swapRB = True)


# In[11]:


Network = cv2.dnn.readNetFromDarknet(confPath, weightPath)
ln = Network.getUnconnectedOutLayersNames()


# In[ ]:





# In[12]:


Network.setInput(Pre_image)
LayerOuts = Network.forward(ln)


# In[13]:


for output in LayerOuts:
    for detection in output:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if float(confidence) > 0.8:
            Object = Labels[classId]
            if Object == "person":
                (centerX, centerY, width, height) = (detection[:4] * np.array([W, H, W, H])).astype("int")
                X, Y = int(centerX - (width/2)), int(centerY - (height/2))
                cv2.rectangle(image, (X, Y), (X+width, Y+height), (0, 0, 255), 2)
                cv2.putText(image, Object, (X, Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
cv2.imwrite("DETECTED_IMAGE.jpg", image)
        


# In[6]:





# In[14]:


from PIL import Image
Image.open("DETECTED_IMAGE.jpg")

