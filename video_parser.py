import numpy as np
import cv2
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

class Video:
	def __init__(self,direc,name,start,end,crop_startx,crop_endx,crop_starty=0,crop_endy=360):
		self.id=name
		self.object=cv2.VideoCapture(direc+name)
		self.start=start
		self.end=end
		self.width=self.object.get(3)	
		self.height=self.object.get(4)	
		self.frame=self.object.get(7)	
		self.fr=self.object.get(5)
		self.crop_startx=crop_startx
		self.crop_endx=crop_endx
		self.crop_starty=crop_starty
		self.crop_endy=crop_endy
	def info(self):
		print self.id
		print "video\t", self.start,":",self.end, "\tcropx\t", self.crop_startx,":",self.crop_endx,"\tcropy\t", self.crop_starty, ":", self.crop_endy
vid_name=["purple_adam.mp4","purple_becky.mp4","purple_ellie.mp4","purple_fern.mp4",\
"purple_hayleighs.mp4","purple_henry.mp4","purple_jo.mp4","purple_jordan.mp4",\
"purple_katherine.mp4","purple_tom.mp4"]
vid_dir='/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_videos/driving_videos/purple/'
vid_name2=["lc_andrea.mp4","lc_andy.mp4","lc_bobbyjo.mp4","lc_bobclark.mp4","lc_claire.mp4",\
"lc_danielle.mp4","lc_dean.mp4","lc_emma.mp4","lc_kev.mp4","lc_nikki.mp4"]
vid_dir2="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_videos/driving_videos/learning_centre/"
video=[]
video2=[]
video.append(Video(vid_dir,vid_name[0],600,6000,60,420))  
video.append(Video(vid_dir,vid_name[1],4610,10010,50,410))  
video.append(Video(vid_dir,vid_name[2],5400,10800,60,420))  
video.append(Video(vid_dir,vid_name[3],11700,17100,0,360))  
video.append(Video(vid_dir,vid_name[4],12600,18000,0,360))  
video.append(Video(vid_dir,vid_name[5],750,5250,30,300,45,315))  
video.append(Video(vid_dir,vid_name[6],3600,9000,0,360))  
video.append(Video(vid_dir,vid_name[7],10800,16200,35,395))  
video.append(Video(vid_dir,vid_name[8],600,6000,0,360))  
video.append(Video(vid_dir,vid_name[9],12150,17550,40,310,45,315)) 
for x in range(10):
	video[x].info()

def process_video(origin_path,store_path, name, start,end,crop_startx,crop_endx,crop_starty,crop_endy):
	cap = cv2.VideoCapture(origin_path+name)
	print "processing",name
	for x in range(end):
		ret, frame = cap.read()
	
		#cv2.imshow("frame",frame)
		if x>=start:
			crop_img=frame[crop_starty:crop_endy, crop_startx:crop_endx] #frame(y1:y2,x1:x2)
			crop_img=cv2.flip(crop_img,1)
			head=name.split(".mp4")[0]
			store_path2=store_path+"crop/"+head+"/"+head+"_"+str(x)+".jpg"
			cv2.imwrite(store_path2,crop_img)
	cap.release()
	cv2.destroyAllWindows()

origin_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_videos/driving_videos/purple/"
store_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_videos/driving_videos_v2/"
origin_path2="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_videos/driving_videos/learning_centre/"
store_path2="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_videos/driving_videos_v2/"

process_video(origin_path2,store_path2,"lc_bobbyjo.mp4",12000,16500,0,360,0,360)

for x in range(10):
	process_video(origin_path,store_path,video[x].id,video[x].start,video[x].end,video[x].crop_startx,video[x].crop_endx,video[x].crop_starty,video[x].crop_endy)
	



#0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
#1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
#2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
#3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
#4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
#5. CV_CAP_PROP_FPS Frame rate.
#6. CV_CAP_PROP_FOURCC 4-character code of codec.
#7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
#8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
#9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
#10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
#11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
#12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
#13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
#14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
#15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
#16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
#17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
#18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

#while(cap.isOpened()):
#    ret, frame = cap.read()
#    
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    #print "log"
#    cv2.imshow('frame',gray)
#    
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#dictionary
#test={"a":1,"b":2}
#test["c"]=3
#print "test",test


#frame 20572.0 width 640.0 height 360.0 fr 29.97002997 name purple_adam.mp4	capture 0:20-3:20, 600:6000, crop 60:420, 
#frame 18499.0 width 640.0 height 360.0 fr 29.970005693 name purple_becky.mp4 capture 2:33-5:33, 4610:10010, crop 50:410
#frame 28382.0 width 640.0 height 360.0 fr 29.97002997 name purple_ellie.mp4 capture 3:00-6:00, 5400:10800, crop 60:420 
#frame 37944.0 width 640.0 height 360.0 fr 29.9700181341 name purple_fern.mp4 capture 6:30-9:30, 11700:17100, crop 0:360
#frame 26952.0 width 640.0 height 360.0 fr 29.97002997 name purple_hayleighs.mp4 capture 7:00-10:00, 12600:18000, crop 0:360
#frame 22668.0 width 480.0 height 360.0 fr 25.0 name purple_henry.mp4 capture 0:30-3:30, 750:5250, crop 30:300, 45:315
#frame 38345.0 width 640.0 height 360.0 fr 29.97002997 name purple_jo.mp4 capture 2:00-5:00, 3600:9000, crop 0:360
#frame 27016.0 width 640.0 height 360.0 fr 29.97002997 name purple_jordan.mp4 capture 6:00-9:00, 10800:16200, crop 35:395
#frame 10041.0 width 640.0 height 360.0 fr 29.9699852433 name purple_katherine.mp4 capture 0:20-3:20, 600:6000, crop 0:360
#frame 23859.0 width 480.0 height 360.0 fr 29.97002997 name purple_tom.mp4 capture 6:45-9:45, 12150:17550, crop 40:310, 45:315
#frame 127833.0 width 640.0 height 360.0 fr 25.0 name lc_andrea.mp4
#frame 135524.0 width 640.0 height 360.0 fr 24.8700745787 name lc_andy.mp4
#frame 110188.0 width 640.0 height 360.0 fr 25.0 name lc_bobbyjo.mp4 capture 8:00-11:00,12000:16500, crop 0:360
#frame 108430.0 width 640.0 height 360.0 fr 25.0 name lc_bobclark.mp4
#frame 87399.0 width 640.0 height 360.0 fr 25.0 name lc_claire.mp4
#frame 88858.0 width 640.0 height 360.0 fr 25.0 name lc_danielle.mp4
#frame 63515.0 width 640.0 height 360.0 fr 25.0 name lc_dean.mp4
#frame 118435.0 width 640.0 height 360.0 fr 25.0 name lc_emma.mp4
#frame 72775.0 width 640.0 height 360.0 fr 25.0 name lc_kev.mp4
#frame 115058.0 width 640.0 height 360.0 fr 25.0 name lc_nikki.mp4


