# INTRODUCTION
The global pandemic COVID-19 situation has emerged as a dangerous disease spreads around the 
world. Wearing a face mask can help prevent the spread of infection and prevent you from contracting 
infectious bacteria in the air. Face mask detection systems can be used to monitor whether people are 
wearing masks.
Here, **the HAAR-CASACADE algorithm** is used for image recognition. Combined with other 
existing algorithms, this classifier produces high recognition rates, efficient feature selection, and a 
small number of false-positive feature selections even when using different representations. 
HAAR's feature-based cascading classification system uses only 200 features out of 6000 to achieve 
85-95% recognition rate.
According to this motivation I think mask detection as a unique and public health service system 
during the global pandemic COVID-19 epidemic. The model is trained by face with mask image 
and face without mask image.
 Keywords: cv2, Haarcascade classifier, etc.
# BACKGROUND AND MOTIVATION
The world has not yet completely recovers from this epidemic and the vaccine that can effectively 
treat covid- 19 is yet to be discovered. Still, to reduce the impact of the epidemic on the country's 
frugality, several governments have allowed a limited number of profitable conditioning to be 
proceeded once the number of new cases of Covid- 19 has dropped below a certain position. As 
those nations carefully restarting their monetary sports, worries have emerged concerning place of 
business protection with inside the new post-Covid-19 environment.


# PROBLEM STATEMENT AND OBJECTIVES
To reduce the possibility of infection, it's advised that people should wear masks and maintain a 
distance of at least 1 cadence from each other. Deep literacy has gained further attention in object 
discovery and was used for mortal discovery purposes and develops a face mask discovery tool 
that can descry whether the existent is wearing mask or not. This can be done by evaluation of the 
bracket results by assaying real- time streaming from the Camera. In deep literacy systems, we 
need a training data set. It's the factual dataset used to train the model for performing colorful 
conduct.
The main aim of the face discovery model is to descry the face of individualities and conclude whether 
they're wearing masks or not at that particular moment when they're captured in the image.       


# RELATED TECHNOLOGY
Machine learning and deep learning have gained in popularity because of the recent push in the AI 
industry, and early adopters of this technology are starting to see benefits. Machine learning and deep 
learning can be implemented using a variety of programming languages, each of which focuses on a 
different aspect of the problem. Python is the most common language for machine learning and deep 
learning. For a long time, python has been preferred programming language for machine learning and deep 
learning researchers. Python provides developers with some of the most flexible and feature-rich tools that 
improve not only their productivity but also the quality of their code. It is one of the most developerfriendly programming languages, with a diverse set of libraries to serve every use case or projects.

# FACE MASK DETECTION PYTHON CODE

import winsound 
frequency = 2500 
duration = 1000 
 
import numpy as np 
import cv2 
import random 
 
face_cascade = cv2.CascadeClassifier('C:\\haar-cascade-files-master\\haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('C:\\haar-cascade-files-master\\haarcascade_eye.xml') 
mouth_cascade = cv2.CascadeClassifier('C:\\haar-cascade-files-master\\haarcascade_mcs_mouth.xml') 
upper_body = cv2.CascadeClassifier('C:\\haarcascade_upperbody.xml') 
 
 
 
**Adjust threshold value in range 80 to 105 based on your light.** 

bw_threshold = 80 
 
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (30, 30) 
weared_mask_font_color = (155, 45, 185) 
not_weared_mask_font_color = (122, 0, 205) 
thickness = 2 
font_scale = 1 
weared_mask = "With MASK" 
not_weared_mask = "No MASK" 
 
**Start Capturing video** 

cap = cv2.VideoCapture(0) 
 
while 1: 
    ret, img = cap.read() 
    img = cv2.flip(img,1) 
 
    # Convert Image into gray 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
    # Convert image in black and white 
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY) 
    #cv2.imshow('black_and_white', black_and_white) 
 
    # detect face 
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
 
    # Face prediction for black and white 
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4) 
 
 
    if(len(faces) == 0 and len(faces_bw) == 0): 
        cv2.putText(img, "No face found...", org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA) 
    elif(len(faces) == 0 and len(faces_bw) == 1): 
        # It has been observed that for white mask covering mouth, with gray image face prediction is not happening 
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA) 
    else: 
        # Draw rectangle on face 
        for (x, y, w, h) in faces: 
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 100, 25), 2) 
            roi_gray = gray[y:y + h, x:x + w] 
            roi_color = img[y:y + h, x:x + w] 
 
 
            # Detect lips counters 
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5) 
 
        # Face detected but Lips not detected which means person is wearing mask 
        if(len(mouth_rects) == 0): 
            cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA) 
        else: 
            for (mx, my, mw, mh) in mouth_rects: 
 
                if(y < my < y + h): 
        # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and person is not waring mask 
                    cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA) 
                    winsound.Beep(frequency, duration) 
                    #cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3) 
                    break 
 
    # Show frame with results 
    cv2.imshow('Mask Detection', img) 
    k = cv2.waitKey(10) & 0xff 
    if k == 'q': 
        break 
 
# Release video 
cap.release() 
cv2.destroyAllWindows()
