#Created by AhmadrezaAnaami
import cv2
import numpy as np

cap = cv2.VideoCapture("RES/2.mp4")


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
face_cascade = cv2.CascadeClassifier('RES/haarcascade_fullbody.xml')


color_list=['red','blue','white']
boundaries = [
    ([17, 15, 75], [50, 56, 200]), 
    ([43, 31, 4], [250, 88, 50]), 
    ([187,169,112],[255,255,255]) 
    ]


idx = 0
while True:
    ret , frame = cap.read()
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_green = np.array([40,40, 40])
    upper_green = np.array([70, 255, 255])

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    lower_red = np.array([0,31,255])
    upper_red = np.array([176,255,255])

    lower_white = np.array([0,0,0])
    upper_white = np.array([0,0,255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    
    
    
    kernel = np.ones((13,13),np.uint8)
    thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV |  cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("window2", thresh)
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
	
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        

        if(h>=(1.5)*w):
            if(w>15 and h>= 15):
                idx = idx+1
                player_img = frame[y:y+h,x:x+w]
                player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)

                mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)

                mask2 = cv2.inRange(player_hsv, lower_white, upper_white)
                res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
                res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
                nzCountred = cv2.countNonZero(res2)

                if(nzCount >= 20):
                    cv2.putText(frame, 'barcelona', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                else:
                    pass
                if(nzCountred>=20):
                    cv2.putText(frame, 'real madrid', (x-2, y-2), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
                else:
                    pass


    
    
    cv2.imshow("window", frame)

    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        break

