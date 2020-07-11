from flask import Flask,render_template,request
import cv2
import numpy as np
import dlib
from math import hypot
import base64
import io
from io import StringIO
from io import BytesIO
from PIL import Image
import sys
import re
from imageio import imread
font=cv2.FONT_HERSHEY_COMPLEX


app=Flask(__name__)
@app.route('/')
def home():
	return render_template('font.html')
data=""
@app.route('/webcame',methods=['POST','GET'])
def webcame():
	if request.method == 'POST':
		user = request.form['data']
		user = user.split(',')[1]
		detecter=dlib.get_frontal_face_detector()
		predict=dlib.shape_predictor("C:/Users/NANDAN/Downloads/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

		def center_point(p1,p2):
			return int((p1.x+p2.x)/2),int((p1.y+p2.y)/2)
		def eye_blinking_ratio(eye_points,eye_landmarks):
		     left_point=(eye_landmarks.part(eye_points[0]).x,eye_landmarks.part(eye_points[0]).y)
		     right_point=(eye_landmarks.part(eye_points[3]).x,eye_landmarks.part(eye_points[3]).y)
		     center_top=center_point(eye_landmarks.part(eye_points[1]),eye_landmarks.part(eye_points[2]))
		     center_bottom=center_point(eye_landmarks.part(eye_points[5]),eye_landmarks.part(eye_points[4]))
		        
		     ver_line=cv2.line(frame,center_top,center_bottom,(0,255,0),3)
		     hor_line=cv2.line(frame,left_point,right_point,(0,255,0),3)
		        
		     length_of_ver_line=hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))
		     length_of_hor_line=hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
		        
		     ratio=length_of_hor_line/length_of_ver_line
		     return ratio

		def create_img_frame(data):
		    decode_data=base64.b64decode(data)
		    buffer=BytesIO()
		    buffer.write(decode_data)
		    pimg=Image.open(buffer)
		    frames=np.array(pimg)
		    return frames


		count=0
		
		frame=create_img_frame(user)
		# cv2.imshow("frame",frame)
		gray_face=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces=detecter(gray_face)
		for face in faces:
			x,y=face.left(),face.top()
			x1,y1=face.right(),face.bottom()
			cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),4)
			landmarks=predict(gray_face,face)
			x=landmarks.part(36).x
			y=landmarks.part(36).y
			cv2.circle(frame,(x,y),5,(0,0,255),4)
			left_eye_ratio=eye_blinking_ratio([36,37,38,39,40,41],landmarks)
			right_eye_ratio=eye_blinking_ratio([42,43,44,45,46,47],landmarks)
			average=(left_eye_ratio+right_eye_ratio)/2
			if average >5.10:
				cv2.putText(frame,"BLINKING",(50,255),7,font,(0,233,0))
				count+=1
			cv2.putText(frame,f"count={count}",(50,255),3,font,(0,233,0))
		return f"{count}"
	# cv2.destroyAllWindows()
	#     
if __name__ == '__main__':
	app.run(debug=True)
	#     