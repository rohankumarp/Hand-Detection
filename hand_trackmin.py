import cv2
import mediapipe as mp
import time
cap= cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils   # to draw the points and lines on hands

#for fps
pTime=0 #previous time
cTime=0 #current time

while True :
    ret , img = cap.read()
    imgrbg = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    result = hands.process(imgrbg)
    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks :
        for handlmrk in result.multi_hand_landmarks :
            #this is giving us the position of each landmark on our hand so that we can highlight a particular landmark
            for id , lm in enumerate(handlmrk.landmark) :
                h , w, c = img.shape  #c means channels that is the color components
                cx , cy = int(lm.x*w) , int(lm.y*h)

                print(id,cx,cy)
                #if id == 0 :
                    #cv2.circle(img,(cx,cy) , 3 ,(255,0,255),25,cv2.FILLED )
            mpDraw.draw_landmarks(img , handlmrk, mpHands.HAND_CONNECTIONS)

    #to find the fps
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #to display the text on screen
    cv2.putText(img , str(int(fps)) , (10,50),cv2.FONT_HERSHEY_COMPLEX,1.6 ,(0,255,255),2)

    cv2.imshow('image' , img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
cv2.destroyAllWindows()
