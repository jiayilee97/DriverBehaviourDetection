import numpy as np
import cv2
import random
import os
import caffe

#create image matrix
def matrixer(img):
	img5=cv2.flip(img,0)
	matrix=np.vstack((img, img5))
	matrix=np.vstack((img5,matrix))
	matrix_flip=cv2.flip(matrix,1)
	matrix=np.hstack((matrix_flip,matrix))
	matrix=np.hstack((matrix,matrix_flip))
	return matrix
	
#rotate matrix
def rotator(img,degree):	
	matrix=matrixer(img)
	rows,cols,chan=matrix.shape
	m3=cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
	rotate = cv2.warpAffine(matrix,m3,(cols,rows))
	crop_img = rotate[480:960, 640:1280]
	return crop_img


#translate matrix
def translator(img,b):
	h,w,c = img.shape
	matrix=matrixer(img)
	rows,cols,chan=matrix.shape
	x=random.randrange(-b,b,1)
	
	y=random.randrange(-b,b,1)
	
	M = np.float32([[1,0,x],[0,1,y]]) #[[1,0, +x direction],[0,1,-y direction]]
#	print "x,y",x,y
	translate = cv2.warpAffine(matrix,M,(cols,rows))
	crop_img = translate[h:2*h, w:2*w]
	return crop_img

#scale, original 640*480
def scaler(img,scale_factor):
	h,w,c = img.shape
	new_h=int(h*scale_factor)
	new_w=new_h*w/h
#	print "new",new_h,new_w

	res = cv2.resize(img,(new_w,new_h)) #(width,height)
	new_x=(new_w-w)/2
	new_y=(new_h-h)/2
	res=res[new_y:new_y+h,new_x:new_x+w] 
	return res


def process_image(img_path,filename,storepath,truth):
#	img=cv2.imread("/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/augment/img_34.jpg")
	img=cv2.imread(img_path+filename)
	cv2.imwrite(storepath+filename,img)
	head,tail=filename.split(".")
#	print "head",head
#	print "tail",tail
#	cv2.imshow("original",img)

	img_flip=cv2.flip(img,1)
#	cv2.imshow("flip",img_flip)
	name_flip=head+"_flip."+tail
	cv2.imwrite(storepath+name_flip,img_flip)
	
	angle=random.randrange(-45,45) #rotation angle
	img_rotate=rotator(img,angle)
#	cv2.imshow("rotate",img_rotate)
	name_rotate=head+"_rotate."+tail
#	print "name_rotate",name_rotate
	cv2.imwrite(storepath+name_rotate,img_rotate)

#	h=random.randrange(-80,80,1) #translation distance
	img_translate=translator(img,80)
#	cv2.imshow("translate",img_translate)
	name_translate=head+"_trans."+tail
	cv2.imwrite(storepath+name_translate,img_translate)

	res=scaler(img,random.uniform(1.0,1.2))
#	cv2.imshow("scale",res)
	name_scale=head+"_scale."+tail
	cv2.imwrite(storepath+name_scale,res)
	
	
	global line
	line.append(filename+" "+truth+"\r\n")
	line.append(name_flip+" "+truth+"\r\n")
	line.append(name_rotate+" "+truth+"\r\n")
	line.append(name_translate+" "+truth+"\r\n")
	line.append(name_scale+" "+truth+"\r\n")
	

	#destroy windows
	#cv2.waitKey(0)
#	cv2.destroyAllWindows()

img_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/imgs/train/"
img_store_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/augment/imgs_aug/"
img_list_train="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/10_cross_validation_list_v3/train_kfold0.txt"
img_list_val="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/10_cross_validation_list_v3/val_kfold0.txt"

caffe.set_mode_gpu();
caffe.set_device(0);

#--- train set
with open(img_list_train,"r") as f:
	y_true=[]
	im_name=[]
	im_names = f.readlines()
	imagecount=len(im_names)
	for i in range(imagecount):
		im_names[i] = im_names[i].strip()#removes leading character
		temp = im_names[i].split(" ")[0]
		im_name.append(im_names[i].split(" ")[0])
		head, tail = os.path.split(temp) #head is not ground truth; head is folder where image is stored, tail is image name
		ground_truth = im_names[i].split(" ")[1]
		y_true.append(int(ground_truth))

txt_train="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/augment/train_aug.txt"
line=[]
for x in range(len(im_name)):#change back to len(im_name)
	if(x%10==0):
		print "processing",x
	process_image(img_path,im_name[x],img_store_path,str(y_true[x]))

with open(txt_train,"w") as g:
	random.shuffle(line)
	g.writelines(line)

#--- val set
with open(img_list_val,"r") as f:
	y_true=[]
	im_name=[]
	im_names = f.readlines()
	imagecount=len(im_names)
	for i in range(imagecount):
		im_names[i] = im_names[i].strip()#removes leading character
		temp = im_names[i].split(" ")[0]
		im_name.append(im_names[i].split(" ")[0])
		head, tail = os.path.split(temp) #head is not ground truth; head is folder where image is stored, tail is image name
		ground_truth = im_names[i].split(" ")[1]
		y_true.append(int(ground_truth))

txt_val="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/augment/val_aug.txt"
line=[]
for x in range(len(im_name)):#change back to len(im_name)
	if(x%10==0):
		print "processing",x
	process_image(img_path,im_name[x],img_store_path,str(y_true[x]))

with open(txt_val,"w") as g:
	random.shuffle(line)
	g.writelines(line)
	
#Image Translation: -7 to 7
#Image Rescaling Factor: 0.8 to 1.2
