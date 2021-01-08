#!/usr/bin/env python
# coding: utf-8

# In[4]:


from PIL import Image


# In[1]:


import numpy as np
import cv2
import os


# In[10]:


Image.open("social Distance.png")


# In[6]:


LabelPath = "coco.names"
weightPath = "yolov3.weights"
confPath = "yolov3.cfg"

Labels = open(LabelPath).read().strip().split("\n")
YoloShape = (416, 416)
Mainimage = cv2.imread("social Distance.png")
H, W = Mainimage.shape[:2]  #### we will use in for rectangle
Pre_image = cv2.dnn.blobFromImage(Mainimage, 1/255.0, YoloShape, swapRB = True)


# In[29]:


Network = cv2.dnn.readNetFromDarknet(confPath, weightPath)
ln = Network.getUnconnectedOutLayersNames()


# In[30]:


Network.setInput(Pre_image)
LayerOuts = Network.forward(ln)


# In[ ]:





# In[31]:


Person = []
X1 = []
Y1 = []
X2 = []
Y2 = []
Cx = []
Cy = []
numP = 0
for output in LayerOuts:
    for detection in output:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if float(confidence) > 0.8:
            Object = Labels[classId]
            if Object == "person":
                numP += 1
                (centerX, centerY, width, height) = (detection[:4] * np.array([W, H, W, H])).astype("int")
                X, Y = int(centerX - (width/2)), int(centerY - (height/2))
                ### Storing out Data (X, Y , Cx, Cy)
                X1.append(X)
                Y1.append(Y)
                X2.append(X + width)
                Y2.append(Y + height)
                Cx.append(centerX)
                Cy.append(centerY)
                Person.append("Person-{}".format(numP))
    
                cv2.rectangle(Mainimage, (X, Y), (X+width, Y+height), (0, 255, 0), 2)
                cv2.putText(Mainimage, "Person-{}".format(numP), (X, Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
cv2.imwrite("DETECTED_IMAGE_new.jpg", Mainimage)
        


# In[32]:


import pandas as pd
df = pd.DataFrame({
    "Person":Person,
    "X1":X1, "Y1":Y1, "X2":X2, 
    "Y2":Y2, "Cx":Cx, "Cy":Cy,
})


# In[33]:


df


# In[34]:


from PIL import Image
Image.open("DETECTED_IMAGE_new.jpg")


# In[19]:


df


# In[35]:


Total_Persons = len(df)
for i in range(Total_Persons):
    for j in range(i+1, Total_Persons):
        distance = np.sqrt((df["Cx"][i] - df["Cx"][j])**2 + (df["Cy"][i] - df["Cy"][j])**2)
        if distance < 100:
            Person1 = df.iloc[i, 1:5].values
            Person2 = df.iloc[j, 1:5].values
            image = cv2.rectangle(Mainimage, tuple(Person1[:2]), tuple(Person1[2:]), (0,0,255), 2)
            image = cv2.rectangle(Mainimage, tuple(Person2[:2]), tuple(Person2[2:]), (0,0,255), 2)
cv2.imwrite("New_Image.jpg", Mainimage)


# In[36]:


from PIL import Image
Image.open("New_Image.jpg")


# In[ ]:




