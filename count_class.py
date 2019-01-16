from morph.centroidtracker import CentroidTracker
from morph.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2

class person_count(object):
	p=str()
	m=str()
	i=str()
	o=str()
	c=float()
	s=int()
	objectID=int(0)
	def __init__(self,p,m,i,o,c,s):
		self.p = p
		self.m = m
		self.i = i
		self.o = o
		self.c = c
		self.s = s
	def track(self):
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
		net = cv2.dnn.readNetFromCaffe(self.p, self.m)
		vs = VideoStream(src=0).start()
		writer = None
		W = None
		H = None
		ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
		ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
		trackers = []
		trackableObjects = {}

		totalFrames = 0
		totalDown = 0
		totalUp = 0
		fps = FPS().start()
		

		while True:
			frame = vs.read()
			if self.i is not None and frame is None:
				break
			frame = imutils.resize(frame, width=500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			if W is None or H is None:
				(H, W) = frame.shape[:2]
			if self.o is not None and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(self.o, fourcc, 30,
					(W, H), True)
			status = "Waiting"
			rects = []
			if totalFrames % self.s == 0:
				status = "Detecting"
				trackers = []
				blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
				net.setInput(blob)
				detections = net.forward()

				for i in np.arange(0, detections.shape[2]):
					
					confidence = detections[0, 0, i, 2]

					
					if confidence > self.c:
						
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
			
			objects = ct.update(rects)

			for (self.objectID, centroid) in objects.items():
				
				to = trackableObjects.get(self.objectID, None)

				if to is None:
					to = TrackableObject(self.objectID, centroid)

				
				else:
					y = [c[1] for c in to.centroids]
					direction = centroid[1] - np.mean(y)
					to.centroids.append(centroid)

					if not to.counted:
						
						if direction < 0 and centroid[1] < H // 2:
							totalUp += 1
							to.counted = True

						
						elif direction > 0 and centroid[1] > H // 2:
							totalDown += 1
							to.counted = True

				
				trackableObjects[self.objectID] = to

				
				
				text = "ID {}".format(self.objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			
			'''info = [
				("Up", totalUp),
				("Down", totalDown),
				("Status", status),
			]

			
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)'''

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
		#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

		if writer is not None:
			writer.release()

		# if we are not using a video file, stop the camera video stream
		#if not get(self.i, False):
		#	vs.stop()

		else:
			vs.release()

		cv2.destroyAllWindows()

	def counting(self):
		print(self.objectID)
