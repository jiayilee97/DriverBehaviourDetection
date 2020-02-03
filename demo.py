import numpy as np
import sys
import os
"""
try:
	caffe_root = os.environ['CAFFE_ROOT'] + '/'
	print caffe_root
except KeyError:
  	raise KeyError("Define CAFFE_ROOT in ~/.bashrc")
print caffe_root
sys.path.insert(1, caffe_root+'python/')
"""
caffe_root = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/caffe-master"
import caffe
import cv2
from py_returnCAMmap import py_returnCAMmap
from py_map2jpg import py_map2jpg
import scipy.io
#import cv2
#import numpy as np

iter_count=0
iter_limit=2
overlap_min_threshold=0.4
rejected=0 #initialise
mask_flag=0

pathname3="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/distracted_driver/"

def mask_process(pathname2,filename2,iter_count):

#	pic = cv2.imread(pathname2+filename2)
#	pic = cv2.resize(pic, (256, 256))
#	head,tail=filename2.split(".")
#	cv2.imwrite(thermal_file_path_single+filename2, pic )
#	print "mask_process done"
#	return 1

	head3,tail3=filename2.split(".")
	pic = cv2.imread(pathname3+head3+"_mask."+tail3) #change
#	print "path",pathname2+head3+"_mask."+tail3
	pic = cv2.resize(pic, (256, 256))
	for i in range(256):
		for j in range(256):
			for k in range(3):
				if pic[i][j][k]>0:
					pic[i][j][0]=255
					pic[i][j][1]=255
					pic[i][j][2]=255
	thermalimg=cv2.imread(pathname2+filename2)#change
	thermalimg=cv2.resize(thermalimg,(256,256))
	#for k: 0,1,2 = blue,green,red
	for i in range(256):
		for j in range(256):
			for k in range(3):
				if thermalimg[i][j][1]<50 and thermalimg[i][j][2]<50:
					thermalimg[i][j][0]=0
					thermalimg[i][j][1]=0
					thermalimg[i][j][2]=0
				else:
					thermalimg[i][j][0]=255
					thermalimg[i][j][1]=255
					thermalimg[i][j][2]=255
	superimposed_img=pic + thermalimg
	overlapped_area=(superimposed_img^pic)^thermalimg
	total_overlap=0
	total_thermalimg=0
	for i in range(256):
		for j in range(256):
			for k in range(3):
				total_overlap=total_overlap+overlapped_area[i][j][k]
				total_thermalimg=total_thermalimg+thermalimg[i][j][k]
	overlap_ratio= float(total_overlap)/float(total_thermalimg)
#	print "overlap_ratio: ", overlap_ratio
	non_overlap=thermalimg-overlapped_area
	pic_original = cv2.imread(imPath+im_name[x]) #change
	pic_original = cv2.resize(pic_original, (256, 256))
	remasked_pic=(~non_overlap)&(pic_original)
	if overlap_ratio<overlap_min_threshold:
#		rejected=rejected+1
#		print "alert!",filename2
		head,tail=filename2.split(".")
#		cv2.imwrite(thermal_file_path_single+head+"_o_."+tail, overlapped_area )
#		cv2.imwrite(thermal_file_path_single+head+"_original_."+tail, pic_original )
#		cv2.imwrite(thermal_file_path_single+head+"_no_."+tail, non_overlap )
		if iter_count<(iter_limit-1):
			cv2.imwrite(thermal_file_path_single+head+"."+tail, remasked_pic )
		return 1
	else:
		return 0

def im2double(im):
	return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def process_image2(pathname,filename,truth,iter_count,store_single_path,store_multi_path,store_anno_path,topNum):
	dir2=pathname+filename
#	print "path", dir2
	image = cv2.imread(dir2) #change
	
#	print "flag3"
#	cv2.imshow("img",image)
#	print image
	image = cv2.resize(image, (256, 256))
	image_anno=image.copy()
	#cv2.waitKey(0)
	# Take center crop.
	center = np.array(image.shape[:2]) / 2.0
	crop = np.tile(center, (1, 2))[0] + np.concatenate([
		-np.array([crop_size, crop_size]) / 2.0,
		np.array([crop_size, crop_size]) / 2.0
	])
	crop = crop.astype(int)
	input_ = image[crop[0]:crop[2], crop[1]:crop[3], :]
	# extract conv features
	net.blobs['data'].reshape(*np.asarray([1,3,crop_size,crop_size])) # run only one image
	net.blobs['data'].data[...][0,:,:,:] = transformer.preprocess('data', input_)
	out = net.forward()
	scores = out['prob']
	activation_lastconv = net.blobs[last_conv].data
	## Class Activation Mapping

#	topNum = 5 # generate heatmap for top X prediction results
	scoresMean = np.mean(scores, axis=0)
	
	ascending_order = np.argsort(scoresMean)
	IDX_category = ascending_order[::-1] # [::-1] to sort in descending order

	curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[IDX_category[:topNum],:]) #prints out 4-numbered CAM map
	curResult = im2double(image)
	#from PIL import Image
	for j in range(topNum):
		# for one image
		curCAMmap_crops = curCAMmapAll[:,:,j]
		curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (256,256))
		curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops),(256,256)) # this line is not doing much
	
		curHeatMap = im2double(curHeatMap)
		curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
		curHeatMap = im2double(image)*0.2+im2double(curHeatMap)*0.7
		

		curHeatMap=curHeatMap*255.0
		rect_img = np.zeros((30,256,3), np.uint8) 
		rect_img2 = np.zeros((30,256,3), np.uint8)
	#masking begins
		if j==0:
			curHeatMap_single=curHeatMap #execute mask_process here
		
		rect_img3 = np.zeros((30,256,3), np.uint8)
#		conf=str(scoresMean[IDX_category[j]])

		
#		cv2.putText(rect_img2, "conf "+conf, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1)		
#		image_anno=np.concatenate((image_anno,rect_img), axis=0)
		
	#masking ends
		conf=str(scoresMean[IDX_category[j]])
		if IDX_category[j]==truth:
			cv2.putText(rect_img, categories[IDX_category[j]], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1)
			cv2.putText(rect_img2, conf, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1)		
			cv2.putText(rect_img3, categories[IDX_category[j]] + " {0:.5f}".format(float(conf)), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1) #cv2.putText(target,text,coordinates,font,fontsize,BGR,fontweight)
		else:
			cv2.putText(rect_img, categories[IDX_category[j]], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1)		
			cv2.putText(rect_img2, conf, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1)		
			cv2.putText(rect_img3, categories[IDX_category[j]] + " {0:.5f}".format(float(conf)), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1) #cv2.putText(target,text,coordinates,font,fontsize,BGR,fontweight)
		curHeatMap=np.concatenate((curHeatMap,rect_img), axis=0)
		curHeatMap=np.concatenate((curHeatMap,rect_img2), axis=0)
		image_anno=np.concatenate((image_anno,rect_img3), axis=0)
		if j==0:
			vis=curHeatMap
		else:
			vis = np.concatenate((vis, curHeatMap), axis=1)
	
		cv2.waitKey(0)
	
	cv2.imwrite(store_single_path+filename, curHeatMap_single )	
	cv2.imwrite(store_multi_path+filename, vis )
	cv2.imwrite(store_anno_path+filename,image_anno)
#	print "i",iter_count
	temp=iter_count
	iter_count=temp+1
#	mask_flag=mask_process(thermal_file_path_single,filename,iter_count)#if image remasked, mask_flag=1
#	if mask_flag==1:
#		mask_flag=0
#		head2,tail2=filename.split(".")
##		print "head2,tail2: ",head2,tail2
##		print "flag1"
#		if iter_count<iter_limit:
#			print "iter ", iter_count
#			process_image2(thermal_file_path_single,filename,truth,iter_count)		
##		print "flag2"
#		return


## Be aware that since Matlab is 1-indexed and column-major, 
## the usual 4 blob dimensions in Matlab are [width, height, channels, num]

## In python the dimensions are [num, channels, width, height]
caffe.set_mode_gpu();
caffe.set_device(0);
model = 'drivernet' #change

if model == 'drivernet': #change
	net_weights = '/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/snapshots_CAM_10class_iter_60000.caffemodel' #change
	net_model = '/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/deploy_CAM_10class.prototxt' #change
	out_layer = 'CAM_fc_driver' #change
	last_conv = 'CAM_conv' #change
	crop_size = 224 #change
elif model == 'alexnet':
	net_weights = 'models/alexnetplusCAM_imagenet.caffemodel'
	net_model = 'models/deploy_alexnetplusCAM_imagenet.prototxt'
	out_layer = 'fc9'
	last_conv = 'conv7'
	crop_size = 227
elif model == 'googlenet':
	net_weights = 'models/imagenet_googlenetCAM_train_iter_120000.caffemodel'
	net_model = 'models/deploy_googlenetCAM.prototxt'
	out_layer = 'CAM_fc'
	crop_size = 224
	last_conv = 'CAM_conv'
else:
	raise Exception('This model is not defined')



# load CAM model and extract features

net = caffe.Net(net_model, net_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([104,117,123]))
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

weights_LR = net.params[out_layer][0].data # get the softmax layer of the network
# shape: [1000, N] N-> depends on the network
"""
for file in os.listdir(snapshot_dir):
	    # print file
	    if model_name in file and file.endswith(".caffemodel"):
	        basename = os.path.splitext(file)[0]
	        iter = int(basename.split("{}_iter_".format(model_name))[1])

	        # print iter
	        if iter > max_iter:
	            max_iter = iter
"""

#--- 10 class testing begins
categories=["c0 safe","c1 texting-r","c2 phone-r","c3 texting-l","c4 phone-l","c5 radio","c6 drinking","c7 reaching behind","c8 hair/makeup","c9 chatting"] #change
imListPath="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/test_normal.txt"
with open(imListPath, 'r') as f:
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

imPath="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/img_normal/Distracted_Driver_Detection_Classified/"
store_single_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/thermal_rcnn_v1/test_on_normal_set/10class_test_norm/thermal_image_single/"
store_multi_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/thermal_rcnn_v1/test_on_normal_set/10class_test_norm/thermal_image_multi/"
store_anno_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/thermal_rcnn_v1/test_on_normal_set/10class_test_norm/annotated_image/"

for x in range(1000): #later change 5 back to imagecount
	print "processing ",x
	iter_count=0
	process_image2(imPath,im_name[x],y_true[x],iter_count,store_single_path,store_multi_path,store_anno_path,5)

