import face_recognition
import os
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import clever_cam_calibration.clevercamcalib as ccc

def undist(vid_path):
    cap = cv2.VideoCapture(vid_path)
 
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    
    out = cv2.VideoWriter('undisted.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    while(True):
    ret, frame = cap.read()
    
    if ret == True: 
        height_or, width_or, depth_or = frame.shape
        frame=ccc.get_undistorted_image(frame,ccc.CLEVER_FISHEYE_CAM_640)
        height_unz, width_unz, depth_unz = frame.shape
        frame=cv2.resize(frame,(0,0), fx=(width_or/width_unz),fy=(height_or/height_unz))
        out.write(frame)
        
        cv2.imshow('frame',frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    else:
        break 
    
    cap.release()
    out.release()
    
    cv2.destroyAllWindows() 

def counting(unz_path):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

    '''if not args.get("input", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)'''

    
    
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(unz_path)

    writer = None

    
    W = None
    H = None

    
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    
    totalFrames = 0
    #totalDown = 0
    #totalUp = 0
    total=0
    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = frame[1]

        '''if args["input"] is not None and frame is None:
            break'''

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        
        #if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('counted.avi', fourcc, 30,(W, H), True)

        
        
        status = "Waiting"
        rects = []

        
        if totalFrames % args["skip_frames"] == 0:
            
            status = "Detecting"
            trackers = []

            
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            
            for i in np.arange(0, detections.shape[2]):
                
                confidence = detections[0, 0, i, 2]
                if confidence > args["confidence"]:
                    
                    idx = int(detections[0, 0, i, 1])

                    
                    if CLASSES[idx] != "person":
                        continue

                    
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    
                    trackers.append(tracker)

        
        else:
            
            for tracker in trackers:
                
                status = "Tracking"

                
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))
        #cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            
            to = trackableObjects.get(objectID, None)

            
            if to is None:
                to = TrackableObject(objectID, centroid)

            
            
            else:
                
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                
                if not to.counted:
                    
                    if direction < 0 and centroid[1] < H // 2:
                        #totalUp += 1
                        total += 1
                        to.counted = True

                    
                    elif direction > 0 and centroid[1] > H // 2:
                        #totalDown += 1
                        total += 1
                        to.counted = True

            
            trackableObjects[objectID] = to

            
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        
        info = [
            #("Up", totalUp),
            #("Down", totalDown),
            ('Total',total),
            ("Status", status),
        ]

        
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            #cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        
        if writer is not None:
            writer.write(frame)

        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        
        totalFrames += 1
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    
    if writer is not None:
        writer.release()

    
    if not args.get("input", False):
        vs.stop()

    
    else:
        vs.release()

    cv2.destroyAllWindows()

def rec(c_path):
    video_capture = cv2.VideoCapture(c_path)

    # Load a sample picture and learn how to recognize it.
    faces_images=[]
    for i in os.listdir('faces/'):
        faces_images.append(face_recognition.load_image_file('faces/'+i))
    known_face_encodings=[]
    for i in faces_images:
        known_face_encodings.append(face_recognition.face_encodings(i)[0])
    known_face_names=[]
    for i in os.listdir('faces/'):
        i=i.split('.')[0]
        known_face_names.append(i)


    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        
        cv2.imshow('Video', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    undist('input.avi')
    counting('undisted.avi')
    rec('counted.avi')