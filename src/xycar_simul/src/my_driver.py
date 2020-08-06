##!/usr/bin/env python
import cv2
import rospy
from std_msgs.msg import Int32MultiArray
import time
import numpy as np

# object which stores info of track-s.mkv
cap = cv2.VideoCapture('/home/seiya/catkin_ws/src/xycar_simul/track-s.mkv')

# FRAMES PER SECOND FOR VIDEO
fps = 30

WIDTH = 640
HEIGHT = 480

pts1 = np.array([[200,290],[440,290],[0,390],[640,390]],np.int32)
pts2 = np.array([[100,0],[540,0],[100,480],[540,480]],np.int32)

def Perspective(image, pts1):
    """
    Preprocess the image and capture a Region of Interest and 
    """
    # Change the color of the image from RGB to gray
    imgGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Blurring
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)
    # Binary Threshold
    imgThresh = cv2.inRange(imgBlur, 135, 255, cv2.THRESH_BINARY) # default: 149, 255
    
    cv2.imshow("Binary Threshold", imgThresh)
    cv2.waitKey(1)

    # Canny Edge Detection
    imgEdge = cv2.Canny(imgGray, 170, 200) # default: 270, 300

    cv2.imshow("canny edge", imgEdge)
    cv2.waitKey(1)

    # Combine the two images
    imgFinal = cv2.add(imgThresh, imgEdge)
    
    cv2.line(image,pt1=tuple(pts1[0]),pt2=tuple(pts1[1]),color=(255,0,0),thickness=2)
    cv2.line(image,pt1=tuple(pts1[1]),pt2=tuple(pts1[3]),color=(255,0,0),thickness=2)
    cv2.line(image,pt1=tuple(pts1[3]),pt2=tuple(pts1[2]),color=(255,0,0),thickness=2)
    cv2.line(image,pt1=tuple(pts1[2]),pt2=tuple(pts1[0]),color=(255,0,0),thickness=2)

    # cv2.line(image,pt1=tuple(pts2[0]),pt2=tuple(pts2[1]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[1]),pt2=tuple(pts2[3]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[3]),pt2=tuple(pts2[2]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[2]),pt2=tuple(pts2[0]),color=(255,0,0),thickness=2)

    Matrix = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
    imgPers = cv2.warpPerspective(imgFinal, Matrix, (WIDTH, HEIGHT))

    imgFinal = cv2.cvtColor(imgPers, cv2.COLOR_GRAY2BGR)
    imgFinalDuplicate = imgFinal.copy()
    cv2.imshow("Test", imgPers)
    cv2.waitKey(1)
    return imgFinal, imgFinalDuplicate

def Histogram(imgFinalDuplicate):
    histogramLane = np.uint8([])
    
    for i in range(WIDTH):
        # ROILane = cv2.rectangle(imgFinalDuplicate,(i,140),(i+1,240),(0,0,255),3)
        # Matrix: rows 100, cols 1 
        ROILane= imgFinalDuplicate[HEIGHT-100:HEIGHT, i]
        # scaling (0 ~ 255) to (0 ~ 1)
        ROILane = cv2.divide(ROILane, 255)
        # summation of white pixels: append the number of white pixels in ROILane to hisogramLane  
        histogramLane = np.append(histogramLane, ROILane.sum(axis=0)[0:1].astype(np.uint8), axis=0)

    return histogramLane


def LaneFinder(imgFinal, histogramLane):
    # find the position of left edge of left lane
    LeftLanePos = np.argmax(histogramLane[:140])

    # find the position of left edge of right lane
    RightLanePos = 500 + np.argmax(histogramLane[500:])

    cv2.line(imgFinal, (LeftLanePos, 0), (LeftLanePos, HEIGHT), color=(0,255,0), thickness=2)
    cv2.line(imgFinal, (RightLanePos, 0), (RightLanePos, HEIGHT), color=(0,255,0), thickness=2)
    return LeftLanePos, RightLanePos

def LaneCenter(imgFinal, LeftLanePos, RightLanePos):
    cv2.putText(imgFinal, "LeftLanePos: "+str(LeftLanePos), (200,50), 0, 1, color=(0,0,255), thickness=2)
    cv2.putText(imgFinal, "RightLanePos: "+str(RightLanePos), (200,100), 0, 1, color=(0,0,255), thickness=2)
    
    laneCenter = (RightLanePos - LeftLanePos)/2 + LeftLanePos
    cv2.putText(imgFinal, "laneCenter: "+str(laneCenter), (200,150), 0, 1, color=(0,0,255), thickness=2)
    frameCenter = WIDTH // 2

    cv2.line(imgFinal, (laneCenter, 0), (laneCenter, HEIGHT), color=(0,255,0), thickness=3)
    cv2.line(imgFinal, (frameCenter, 0), (frameCenter, HEIGHT), color=(255,0,0), thickness=3)

    Result = frameCenter - laneCenter
    cv2.putText(imgFinal, "Result: "+str(Result), (200,200), 0, 1, color=(0,0,255), thickness=2)
    cv2.imshow("position of white lanes", imgFinal)
    cv2.waitKey(1)
    return Result

def pub_motor(Angle, Speed):
    drive_info = [Angle, Speed]
    drive_info = Int32MultiArray(data = drive_info)
    pub.publish(drive_info)
 
def start():
    global pub
    rospy.init_node('my_driver')
    pub = rospy.Publisher('xycar_motor_msg', Int32MultiArray, queue_size=1)
    rate = rospy.Rate(30)
    Speed = 5

    Angle = ""

    while True:
        # Read the video file.
        ret, frame = cap.read()

        # If we got frames, show them.
        if ret == True:
            imgFinal, imgFinalDuplicate = Perspective(frame, pts1)
            histogramLane = Histogram(imgFinalDuplicate)
            LeftLanePos, RightLanePos = LaneFinder(imgFinal, histogramLane)
            Result = LaneCenter(imgFinal, LeftLanePos, RightLanePos)
            
            if Result == 0:
                Angle = 0
            
            elif 0 < Result < 5:
                Angle = 5

            elif 5 <= Result < 10:
                Angle = 12

            elif 10 <= Result < 15:
                Angle = 17

            elif 15 <= Result:
                Angle = 20

            elif -5 < Result < 0:
                Angle = -5

            elif -10 < Result <= -5:
                Angle = -12

            elif -15 < Result <= -10:
                Angle = -17

            elif Result <= -15:
                Angle = -20

            # Display the frame at same frame rate of recording
            # Watch lecture video for full explanation
            time.sleep(1/fps)
            cv2.imshow('frame',frame)

            # Press q to quit
            if cv2.waitKey(25) & 0xFF == ord('q'): 
            # if cv2.waitKey(25) == ord('q'):
                break

        # Or automatically break this whole loop if the video is over.
        else:
            break

        pub_motor(Angle, Speed) 
        rate.sleep()
    
    cap.release()

if __name__ == '__main__':

    start()
