
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from random import shuffle
import time
import cv2
import caffe
import logging
from py_returnCAMmap import py_returnCAMmap
from py_map2jpg import py_map2jpg
import scipy.io
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
import datetime

resultsPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/5june2018halfswap/results_merge_cc/"#change
wrong_num=0
testc0_correct=[]
scores_array=[]

def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
def process_image2(pathname,filename,truth,iter_count,store_single_path,store_multi_path,store_anno_path,topNum):
    dir2=pathname+filename
    global img_dim

    image = cv2.imread(dir2) #change
    image = cv2.resize(image, (img_dim, img_dim))
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
    net2.blobs['data'].reshape(*np.asarray([1,3,crop_size,crop_size])) # run only one image
    net2.blobs['data'].data[...][0,:,:,:] = transformer2.preprocess('data', input_)
    out = net2.forward()
    scores = out['prob']
    activation_lastconv = net2.blobs[last_conv].data
    ## Class Activation Mapping

#    topNum = 5 # generate heatmap for top X prediction results
    scoresMean = np.mean(scores, axis=0)
    
    ascending_order = np.argsort(scoresMean)
    IDX_category = ascending_order[::-1] # [::-1] to sort in descending order

    curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[IDX_category[:topNum],:]) #prints out 4-numbered CAM map
    curResult = im2double(image)
    #from PIL import Image
    for j in range(topNum):
        # for one image
        curCAMmap_crops = curCAMmapAll[:,:,j]
        curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (img_dim,img_dim))
        curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops),(img_dim,img_dim)) # this line is not doing much
    
        curHeatMap = im2double(curHeatMap)
        curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
        curHeatMap = im2double(image)*0.2+im2double(curHeatMap)*0.7
        

        curHeatMap=curHeatMap*255.0
        rect_img = np.zeros((30,img_dim,3), np.uint8) 
        rect_img2 = np.zeros((30,img_dim,3), np.uint8)
    #masking begins
        if j==0:
            curHeatMap_single=curHeatMap #execute mask_process here
        
        rect_img3 = np.zeros((30,img_dim,3), np.uint8)
        
    #masking ends
        conf=str(scoresMean[IDX_category[j]])
        
        if IDX_category[j]==truth:
            cv2.putText(rect_img, CLASSES[IDX_category[j]], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1)
            cv2.putText(rect_img2, conf, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1)        
            cv2.putText(rect_img3, CLASSES[IDX_category[j]] + " {0:.5f}".format(float(conf)), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1) #cv2.putText(target,text,coordinates,font,fontsize,BGR,fontweight)
        else:
            cv2.putText(rect_img, CLASSES[IDX_category[j]], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1)        
            cv2.putText(rect_img2, conf, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1)        
            cv2.putText(rect_img3, CLASSES[IDX_category[j]] + " {0:.5f}".format(float(conf)), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1) #cv2.putText(target,text,coordinates,font,fontsize,BGR,fontweight)
        curHeatMap=np.concatenate((curHeatMap,rect_img), axis=0)
        curHeatMap=np.concatenate((curHeatMap,rect_img2), axis=0)
        image_anno=np.concatenate((image_anno,rect_img3), axis=0)
        if j==0:
            vis=curHeatMap
        else:
            vis = np.concatenate((vis, curHeatMap), axis=1)
        cv2.waitKey(0)
    
    global test
#    print "global variable",test
    if "norm" not in test:
        directory=(store_anno_path+filename).split("img")[0]
    else:
        print "norm"
        directory=store_anno_path
#    print("dir",directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
#    print("storepath_multi",store_multi_path+filename)
    if "norm" not in test:
        directory=(store_multi_path+filename).split("img")[0]
        directory_correct=(store_multi_path+"correct/"+filename).split("img")[0]
        directory_wrong=(store_multi_path+"wrong/"+filename).split("img")[0]
    else:
        directory=store_multi_path
        directory_correct=store_multi_path+"correct/"
        directory_wrong=store_multi_path+"wrong/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_correct):
        os.makedirs(directory_correct)
    if not os.path.exists(directory_wrong):
        os.makedirs(directory_wrong)
    cv2.imwrite(store_multi_path+filename, vis )
    cv2.imwrite(store_anno_path+filename,image_anno)
    global wrong_num
    wrong_num=wrong_num+1
#    print "wrong written",wrong_num
    temp=iter_count
    iter_count=temp+1
    if IDX_category[j]==truth:
#        print "store correct:",directory_correct
        cv2.imwrite(store_multi_path+"correct/"+filename, vis )
    else:
#        print "store wrong:",directory_wrong
        cv2.imwrite(store_multi_path+"wrong/"+filename, vis )

CAM_predictions=[]
import random

def process_video(pathname,filename,truth,iter_count,store_single_path,store_multi_path,store_anno_path,topNum,img_dim=224,img=np.zeros((256,256,3)),testc0=0):
    dir2=pathname+filename
#    global img_dim
    print "dir2:",dir2
#    image = cv2.imread(dir2) #change
    
    print "img",img.shape
    image=img
    image = cv2.resize(image, (img_dim, img_dim))
    image_anno=image.copy()
    #cv2.waitKey(0)
    # Take center crop.
    center = np.array(image.shape[:2]) / 2.0
    print"center",center
    print "tile",np.tile(center, (1, 2))[0]
    print"np conc",np.concatenate([
        -np.array([crop_size, crop_size]) / 2.0,
        np.array([crop_size, crop_size]) / 2.0
    ])
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -np.array([crop_size, crop_size]) / 2.0,
        np.array([crop_size, crop_size]) / 2.0
    ])
    print "crop",crop
    crop = crop.astype(int)
    print "crop_size",crop_size
    input_ = image[crop[0]:crop[2], crop[1]:crop[3], :]
    # extract conv features
    net2.blobs['data'].reshape(*np.asarray([1,3,crop_size,crop_size])) # run only one image
    net2.blobs['data'].data[...][0,:,:,:] = transformer2.preprocess('data', input_)
    print "blob dimension",net2.blobs['data'].data[...].shape
    out = net2.forward()
    scores = out['prob']
    activation_lastconv = net2.blobs[last_conv].data
    ## Class Activation Mapping

#    topNum = 5 # generate heatmap for top X prediction results
    scoresMean = np.mean(scores, axis=0)
    
    ascending_order = np.argsort(scoresMean)
    IDX_category = ascending_order[::-1] # [::-1] to sort in descending order

    curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[IDX_category[:topNum],:]) #prints out 4-numbered CAM map
    curResult = im2double(image)
    #from PIL import Image
    for j in range(topNum):
        # for one image
        curCAMmap_crops = curCAMmapAll[:,:,j]
        curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (img_dim,img_dim))
        curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops),(img_dim,img_dim)) # this line is not doing much
    
        curHeatMap = im2double(curHeatMap)
        curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
        curHeatMap = im2double(image)*0.2+im2double(curHeatMap)*0.7
        

        curHeatMap=curHeatMap*255.0
        rect_img = np.zeros((30,img_dim,3), np.uint8) 
        rect_img2 = np.zeros((30,img_dim,3), np.uint8)
    #masking begins
        if j==0:
            curHeatMap_single=curHeatMap #execute mask_process here
        
        rect_img3 = np.zeros((30,img_dim,3), np.uint8)
        
    #masking ends
        conf=str(scoresMean[IDX_category[j]])
        
        if IDX_category[j]==truth:
            cv2.putText(rect_img, CLASSES[IDX_category[j]], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1)
            cv2.putText(rect_img2, conf, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1)        
            cv2.putText(rect_img3, CLASSES[IDX_category[j]] + " {0:.5f}".format(float(conf)), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1) #cv2.putText(target,text,coordinates,font,fontsize,BGR,fontweight)
        else:
            cv2.putText(rect_img, CLASSES[IDX_category[j]], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1)        
            cv2.putText(rect_img2, conf, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1)        
            cv2.putText(rect_img3, CLASSES[IDX_category[j]] + " {0:.5f}".format(float(conf)), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1) #cv2.putText(target,text,coordinates,font,fontsize,BGR,fontweight)
        curHeatMap=np.concatenate((curHeatMap,rect_img), axis=0)
        curHeatMap=np.concatenate((curHeatMap,rect_img2), axis=0)
        image_anno=np.concatenate((image_anno,rect_img3), axis=0)
        if j==0:
            vis=curHeatMap
        else:
            vis = np.concatenate((vis, curHeatMap), axis=1)
        cv2.waitKey(0)

    global CAM_predictions
    CAM_predictions.append(IDX_category[0])
#    CAM_predictions.append(str(random.randint(0,6)))
    print "CAM predictions"," ".join(np.array(CAM_predictions[-31:]).astype(str))
    
    directory=[]
    print "testc0",testc0
    if testc0==0:
        directory.append(store_multi_path+"/".join(filename.split("/")[0:2]))
        directory.append(store_anno_path+"/".join(filename.split("/")[0:2]))
        directory.append(store_multi_path+"correct/"+"/".join(filename.split("/")[0:2]))
        directory.append(store_multi_path+"wrong/"+"/".join(filename.split("/")[0:2]))
    elif testc0==1:
        directory.append(store_multi_path+"/".join(filename.split("/")[0:1]))
        directory.append(store_anno_path+"/".join(filename.split("/")[0:1]))
        directory.append(store_multi_path+"correct/"+"/".join(filename.split("/")[0:1]))
        directory.append(store_multi_path+"wrong/"+"/".join(filename.split("/")[0:1]))

    for item in directory:    
        if not os.path.exists(item):
            os.makedirs(item)
            print "created dir: ",item
    cv2.imwrite(store_multi_path+filename, vis )
    cv2.imwrite(store_anno_path+filename,image_anno)
    if IDX_category[0]==truth:
        cv2.imwrite(store_multi_path+"correct/"+filename,vis)
    else:
        cv2.imwrite(store_multi_path+"wrong/"+filename,vis)

def get_labelname(labelmap, labels):
    # print labelmap
    num_labels = len(labelmap)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == i:
                found = True
                labelnames.append(labelmap[i])
                break
        assert found == True
    return labelnames


class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
    ### RESIZE 256x340 as in training steps
        self.net = caffe.Classifier(model_def,model_weights,image_dims=(256,256),
                                    raw_scale=255,mean=np.array([104, 117, 123]),channel_swap=(2,1,0))
        # self.net = caffe.Net(model_def,  # defines the structure of the model
        #                      model_weights,  # contains the trained weights
        #                      caffe.TEST)  # use test mode (e.g., don't perform dropout)
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape}) #yes
        self.transformer.set_transpose('data', (2, 0, 1)) #yes
        # self.transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        # self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        # self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        # file = open(labelmap_file, 'r')
        # self.labelmap = caffe_pb2.LabelMap()
        self.labelmap = CLASSES
        # print self.labelmap
        # text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.0, topn=2):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        # self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        # Run the net and examine the top_k results
        # transformed_image = self.transformer.preprocess('data', image)
        # self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        scores = self.net.predict([image]).flatten()
        #print "scores is {}".format(scores)
        global scores_array
        scores_array.append(scores)
        #print "scores_array",scores_array
        #print "scores_array[0]",scores_array[0]
        # detections = self.net.forward()['prob']
        # print detections[0]
        # Parse the outputs.

        det_conf = scores

        # Get detections with confidence higher than 0.6.
        conf_arr = np.array(det_conf)
        sorted_inds = np.argsort(-conf_arr)
        # top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
        top_indices = sorted_inds[:3]  # take top 3
        #print top_indices
        top_label_indices = top_indices.tolist()

        top_labels = get_labelname(self.labelmap, top_label_indices)
        #print top_labels
        top_conf = []
        for i in top_indices:
            top_conf.append(conf_arr[i])
        return top_labels, top_conf,top_label_indices
    def detect2(self,image):
#        print "detect2"
        scores = self.net.predict([image]).flatten()
        det_conf = scores
        conf_arr = np.array(det_conf)
        sorted_inds = np.argsort(-conf_arr)
        top_indices = sorted_inds[:2]
        top_label_indices = top_indices.tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_conf = []
        for i in top_indices:
            top_conf.append(conf_arr[i])
        return top_labels, top_conf,top_label_indices
    def detect3(self,image):
        scores = self.net.predict([image]).flatten()
        #added for CAM heatmap
        activation_lastconv = self.net.blobs['CAM_conv'].data
        scoresMean = np.mean(scores, axis=0)
        ascending_order = np.argsort(scoresMean)
        IDX_category = ascending_order[::-1] # [::-1] to sort in descending order  
        curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[IDX_category[:2],:])
        curResult = im2double(image)
        for j in range(topNum):
            curCAMmap_crops = curCAMmapAll[:,:,j]
            curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (img_dim,img_dim))
            curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops),(img_dim,img_dim))
            curHeatMap = im2double(curHeatMap)
            curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
            curHeatMap = im2double(image)*0.2+im2double(curHeatMap)*0.7
            curHeatMap=curHeatMap*255.0
            rect_img = np.zeros((30,img_dim,3), np.uint8) 
            rect_img2 = np.zeros((30,img_dim,3), np.uint8)

            if j==0:
                curHeatMap_single=curHeatMap 
            rect_img3 = np.zeros((30,img_dim,3), np.uint8)
            conf=str(scoresMean[IDX_category[j]])
            
            if IDX_category[j]==truth:
                cv2.putText(rect_img, CLASSES[IDX_category[j]], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1)
                cv2.putText(rect_img2, conf, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1)        
                cv2.putText(rect_img3, CLASSES[IDX_category[j]] + " {0:.5f}".format(float(conf)), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 1) 
            else:
                cv2.putText(rect_img, CLASSES[IDX_category[j]], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1)        
                cv2.putText(rect_img2, conf, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1)        
                cv2.putText(rect_img3, CLASSES[IDX_category[j]] + " {0:.5f}".format(float(conf)), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1) 
            curHeatMap=np.concatenate((curHeatMap,rect_img), axis=0)
            curHeatMap=np.concatenate((curHeatMap,rect_img2), axis=0)
            image_anno=np.concatenate((image_anno,rect_img3), axis=0)
            if j==0:
                vis=curHeatMap
            else:
                vis = np.concatenate((vis, curHeatMap), axis=1)
        
        det_conf = scores
        conf_arr = np.array(det_conf)
        sorted_inds = np.argsort(-conf_arr)
        top_indices = sorted_inds[:2]
        top_label_indices = top_indices.tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_conf = []
        for i in top_indices:
            top_conf.append(conf_arr[i])
        return top_labels, top_conf,top_label_indices

def main(args,model_name,snapshot_dir,txt,store_single_path,store_multi_path,store_anno_path,txt_wrong,txt_correct):
    '''main '''
    if not os.path.isdir(resultsPath):
        os.mkdir(resultsPath)
    global testc0_correct
    max_iter = 0
    index = 0
    wrong_pred = 0
    y_pred = []
    y_true = []
    wrong_im_name=[]
    
    for file in os.listdir(snapshot_dir):
        # print file
        
        if model_name in file and file.endswith(".caffemodel"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(model_name))[1])

            # print iter
            if iter > max_iter:
                max_iter = iter
    # print max_iter
    model_weights = "{}/{}_iter_{}.caffemodel".format(snapshot_dir, model_name, max_iter)

    detection = CaffeDetection(args.gpu_id,
                               args.model_def, model_weights,
                               args.image_resize)
                               
    print "imListPath",imListPath
    with open(imListPath, 'r') as f:
        im_names = f.readlines()
        print "total images",len(im_names)
        for i in range(len(im_names)): #change back to len(im_names)
        # for i in range(10):
#            print "checkpointhere", im_names[i]
            im_names[i] = im_names[i].strip()
            im_name = im_names[i].split(" ")[0]
            head, tail = os.path.split(im_name)
#            print "head",im_name
#            print "path",error_analysis_path+im_name
            ground_truth = im_names[i].split(" ")[1]
            y_true.append(int(ground_truth))
            #print "groundtruth is {}".format(ground_truth)
            im_path = os.path.join(imPath, im_name)
            
            # print im_path, im_name
            name = im_name.split("/")[-1]
            # name = name.split(".")[0]
            save_name = name + "_result.jpg"
            save_path = os.path.join(resultsPath, tail)
            #print save_path
            class_name, conf,top_id = detection.detect(im_path)
            y_pred.append(int(top_id[0]))

            img = Image.open(im_path)
            draw = ImageDraw.Draw(img)
            width, height = img.size
            draw.rectangle(((0, 0), (240, 75)), fill="black")

            for j in range(len(conf)):
                #print "class is {}, gt is {}".format(top_id[i],ground_truth)
                dim = 5 + 20 * (j)

                #font = ImageFont.truetype('C:\\Windows\\Fonts\\Arial.ttf', 20)
                font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 20)
                if int(top_id[j])==int(ground_truth):
                    #print "with in top 3"
                    draw.text((10, dim), "{}. {}: {:.3f}".format(j+1,class_name[j], conf[j]), (0, 255, 0), font=font)
                else:
                    draw.text((10, dim), "{}. {}: {:.3f}".format(j+1,class_name[j], conf[j]), (255, 0, 0), font=font)
            
            if top_id[0] != int(ground_truth):
                    #print "wrong prediction, {} predicted as {}\n".format(ground_truth,top_id[0])
                wrong_pred +=1
                print "wrong {} / {} images ".format(wrong_pred,len(im_names))
                
                wrong_im_name.append(im_name+" "+ground_truth+"\r\n")
#                print "wronged"
            else:
                testc0_correct.append(im_name+" "+ground_truth+"\r\n")
#                print "testc0_correct len:",len(testc0_correct)
            
            iter_count=0
            process_image2(imPath+"/",im_name,int(ground_truth),iter_count,store_single_path,store_multi_path,store_anno_path,2)
            
    #        print "imlistpath",imListPath
            
    #        if i%2 == 0: print "hi"
            if i%100 == 0:
                print "processed {} images".format(i)
        
            #pbar.update(i+1)
        #pbar.finish()    
        print("\n")
        print("using caffemodel file %s " % model_weights)
        with open(txt_wrong,"w") as f:
            f.writelines(wrong_im_name)
        with open(txt_correct,"w") as f:
            f.writelines(testc0_correct)
        print(len(y_true))
        print(len(y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=CLASSES))
        #print "y_true", y_true
        #print "y_pred",y_pred
        #in confusion matrix,y-axis is y_true in alphabetical order,x-axis is y_pred in alphabetical order
        # print the results into file and console
        #model_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/snapshot/solver_data_aug_v2_iter_60000.caffemodel"
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',filename=txt,filemode='w')
        print("caffemodel file: %s" % model_weights)#same
        print("caffemodel weights: %s" % model_weights)#same
        print("results saved to: %s" % txt)
        print("deploy txt: %s" % deploy_txt)
        print("testing list: %s" % imListPath) #imListPath
        print("total images: %d" % len(im_names)) #len(im_names)
        print(confusion_matrix(y_true, y_pred)) #same
        print(classification_report(y_true, y_pred, target_names=CLASSES)) #same
        print("accuracy: %.2f" % accuracy_score(y_true, y_pred))
        #print "flag"
        global scores_array

        logging.info("caffemodel file: %s" % model_weights)
        logging.info("testing list: %s" % imListPath)
        logging.info("total images: %d" % len(im_names))
        logging.info(confusion_matrix(y_true, y_pred))
        logging.info(classification_report(y_true, y_pred, target_names=CLASSES))
        logging.info("accuracy: %.2f" % accuracy_score(y_true, y_pred))
        print("logloss: %.2f" % log_loss(y_true, scores_array))
        logging.info("logloss: %.2f" % log_loss(y_true, scores_array))




def error_grouper(txt_wrong):
    txt_stats=txt_wrong.split(".txt")[0]+"_stats.txt"
    print "txt_wrong",txt_wrong
    print "txt_stats",txt_stats
    with open(txt_wrong,"r") as f:
        im_names=f.readlines()
    c=[0]*10
    for item in im_names:
        for i in range(10):
            if item.startswith("c"+str(i)):
    #            print "c"+str(i)
                c[i]=c[i]+1            
    print c
    stats=["num of wrong predictions:\r\n"]
    for x in range(len(c)):
        stats.append("c"+str(x)+": "+str(c[x])+"\r\n")
    stats.append("sum:"+str(sum(c)))
    print stats
    with open(txt_stats,"w") as g:
        g.writelines(stats)

def detect_sorted_video2(args,model_weights):
    with open("/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180703_sorted_driving_videos/videos_info_v4.txt","r") as f:
        txt=f.readlines()
    video_group={}
    all_clips_stats=[]
    all_clips_stats_float=[]
    for x in range(len(txt)):#rmb to change the range to start from 0, end at len(txt)
        
        txt[x]=txt[x].split(" first")[0]
        
    #print txt
    detection = CaffeDetection(args.gpu_id,args.model_def, model_weights,args.image_resize)
    frame_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180703_sorted_driving_videos/sorted_videos/"
    global img_dim
    date1=datetime.date.today()-datetime.timedelta(days=1)
    date1=str(date1).replace("-","")
    store_single_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180703_sorted_driving_videos/error_analysis/"+date1+"_v4_model20/single/"
    store_multi_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180703_sorted_driving_videos/error_analysis/"+date1+"_v4_model20/multi/"
    store_anno_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180703_sorted_driving_videos/error_analysis/"+date1+"_v4_model20/anno/"
    store_tested_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180703_sorted_driving_videos/tested_img/"+date1+"/"
    iter_count=0
    
    for x in range(len(txt)):
        
        if "abnormal" in txt[x]:
            truth=1
        else:
            truth=0
        num_of_wrong=0
        label_array2=[]
        for root,dir2,file2 in os.walk(frame_dir+txt[x]):
            sorted_file2= sorted(file2)
#            print sorted_file2[0], sorted_file2[-1], len(sorted_file2)
            k=31
            
            total_evaluate=len(sorted_file2)-k
            for a in range(len(sorted_file2)):
    #            print sorted_file2[1:5]
                img=cv2.imread(frame_dir+txt[x]+"/"+sorted_file2[a])
                img=cv2.resize(img,(img_dim,img_dim))
                top_labels, top_conf,top_label_indices=detection.detect2(frame_dir+txt[x]+"/"+sorted_file2[a])
                print txt[x]+"/"+sorted_file2[a]
                #frame_dir,txt[x]+sorted_file2[a],truth,iter_count
                process_video(frame_dir,txt[x]+"/"+sorted_file2[a],truth,iter_count,store_single_path,store_multi_path,store_anno_path,2,img=img)

                label_array2.append(top_label_indices[0])
#                print label_array2[1:40]
                rect_img = np.zeros((320,img_dim,3), np.uint8)
                cv2.putText(rect_img, "instant predict: "+str(top_labels[0]), (5, 80), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
#                img=cv2.resize(img,(360,360))
#                rect_img6 = np.zeros((60,img_dim,3), np.uint8)
                cv2.putText(rect_img, "frame: "+str(a), (5, 50), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, txt[x], (5, 20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
#                img=np.concatenate((img,rect_img6), axis=0)
#                img=np.concatenate((img,rect_img), axis=0)
#                print "resize"
                full_label=""
                full_label2=""
                
                if a<k :
                    for num in label_array2:
                        full_label2=full_label2+" "+str(num)
#                        label_int.append(num)
                    label_ave=str(float(sum(map(int,full_label2.split(' ')[1:])))/(a+1))
                    rounded_ave="{0:.0f}".format(float(label_ave))
                if a>=k:
                    for num in label_array2[-(k+1):-1]:
                        full_label2=full_label2+" "+str(num)
                    label_ave=str(float(sum(map(int,full_label2.split(' ')[1:])))/k)
                    rounded_ave="{0:.0f}".format(float(label_ave))
                    print "full_label2",full_label2
                    #print "rounded_ave prediction:",rounded_ave
                    if int(rounded_ave) !=truth:
                        num_of_wrong=num_of_wrong+1
                        
                cv2.putText(rect_img, "last {} labels: ".format(str(k)) ,(5, 110), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, full_label2[1:21], (5, 125), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, full_label2[21:41], (5, 140), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, full_label2[41:61], (5, 155), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, full_label2[61:87], (5, 170), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, "last {} labels ave: ".format(str(k))+label_ave , (5, 190), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, "rounded ave: " + rounded_ave , (5, 220), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, "temporal predict: " + CLASSES[int(rounded_ave)], (5, 250), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, "num of wrong: "+ str(num_of_wrong) +"/"+str(total_evaluate), (5, 280), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                cv2.putText(rect_img, "accuracy: "+ str(1-float(num_of_wrong)/total_evaluate), (5, 310), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)
                img=np.concatenate((img,rect_img), axis=0)
                store_tested_dir=store_tested_path+txt[x]+"/"
#                print "store tested in dir",store_tested_dir
                if not os.path.exists(store_tested_dir):
                    os.makedirs(store_tested_dir)
                    print "created dir:",store_tested_dir
                print "store tested in",store_tested_path+txt[x]+"/"+sorted_file2[a]+"\n"
                cv2.imwrite(store_tested_path+txt[x]+"/"+sorted_file2[a],img)
#                cv2.imshow("img",img)
#                cv2.waitKey(5)
#            print img.shape
            clip_accuracy=1-float(num_of_wrong)/total_evaluate
            
#            np.array(all_clip_stats).astype(np.float)
#            print "accuracy",str(clip_accuracy)
            all_clips_stats_float.append(clip_accuracy)
            all_clips_stats.append(txt[x]+" "+str(clip_accuracy)+"\r\n")
#            print type(clip_accuracy)
            ave_clip_accuracy=np.mean(all_clips_stats_float)
            print "ave_clip_accuracy",ave_clip_accuracy
            which_group= "_".join(txt[x].split("_")[0:2])
            if which_group not in video_group:
                print "yes"
                video_group[which_group]=[]
                video_group[which_group].append(clip_accuracy)
            else:
                video_group[which_group].append(clip_accuracy)
            
            stats_store_path="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180703_sorted_driving_videos/"+date1+"_clips_stats_model20_v4.txt"
            with open(stats_store_path,"w") as f:
                
                f.writelines(all_clips_stats)
                f.writelines("\r\n"+"overall mean accuracy of all videos: "+str(ave_clip_accuracy)+"\r\n")
                f.writelines("mean accuracy of each video: \r\n")
                f.writelines("\r\nvideogroup:\r\n"+str(video_group)+"\r\n")
                for item in sorted(video_group):
                    f.writelines(item+" "+str(np.mean(video_group[item]))+"\r\n")
                
            print "videogroup",video_group
            print "saved in", stats_store_path
#            print all_clips_stats
            

def parse_args(deploy_txt):
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    # parser.add_argument('--labelmap_file',
    #                     default='examples/distracted_driver/labelmap_driver.prototxt')
    parser.add_argument('--model_def',
                        default="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/CAM-Python_HaSon/models/2classes_deploy_driver.prototxt")#change
    parser.add_argument('--image_resize', default=224, type=int)
    parser.add_argument('--testing', default="ori_cropped_12515", type=str)
    parser.add_argument('--use_model', default="model21", type=str)

    return parser.parse_args()



CLASSES = ["safe","unsafe"]
deploy_txt='/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/CAM-Python_HaSon/models/2classes_deploy_driver.prototxt'

target_model=parse_args(deploy_txt).use_model
if target_model=="model8":
    model_name="snapshots_CAM_2class_excludeCxtoCy"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/excludeCxtoCy/snapshot_v2"
elif target_model=="model5":
    model_name="snapshots_CAM_2class"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask"
elif target_model=="model9":
    model_name="snapshots_CAM_2class_trainExcludeC2toC5"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/excludeCxtoCy/snapshot_v3"
elif target_model=="model14":
    model_name="snapshots_CAM_2class_trainExcludeC1C6toC7C8"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/excludeCxtoCy/excludeC1C6toC7C8/snapshot"
elif target_model=="model15":
    model_name="snapshots_CAM_2class_ttransnorm"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180711_masked_norm/snapshot"
elif target_model=="model16":
    model_name="snapshots_CAM_2class_ttransnorm"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180711_masked_norm/snapshot_v2"
elif target_model=="model17":
    model_name="snapshots_CAM_2class_ttransnorm"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180711_masked_norm/snapshot_v3"
elif target_model=="model18":
    model_name="snapshots_CAM_2class_ttransnorm"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180711_masked_norm/snapshot_v4"
elif target_model=="model19":
    model_name="snapshots_CAM_2class"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/snapshot/v1"
elif target_model=="model20":
    model_name="snapshots_CAM_2class"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/snapshot/v2_model20"
    root_error_analysis="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180703_sorted_driving_videos/testc0/error_analysis/"
    version="_v3_test_correct_only"
elif target_model=="model21":#change
    model_name="snapshots_CAM_2class"
    snapshot_dir="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/snapshot/v4_model21"
    root_error_analysis="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/error_analysis/"
    version="_v1"

    
test=parse_args(deploy_txt).testing

if test=="ori":       
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/excludeCxtoCy/test_2class_excludeCxtoCy_v2.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/caffe-master/data/imgs/train"
elif test=="rcnn":
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/excludeCxtoCy/test_2class_excludeCxtoCy_v2.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/img_rcnn_v1"
elif test=="norm":
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/test_normal.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/img_normal/Distracted_Driver_Detection_Classified"
elif test=="norm_rcnn_partial":
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180711_masked_norm/masked_norm_test.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180711_masked_norm/masked"
elif test=="ori_cropped":       
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/excludeCxtoCy/test_2class_excludeCxtoCy_v2.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/cropped_img/ori"
elif test=="rcnn_cropped":
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/excludeCxtoCy/test_2class_excludeCxtoCy_v2.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/cropped_img/rcnn"
elif test=="norm_cropped":
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/masked_norm.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/cropped_img/norm"
elif test=="norm_rcnn_cropped":
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/masked_norm.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/cropped_img/norm_rcnn"
elif test=="train_ori_cropped":
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/train_2class_excludeCxtoCy_v2.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/cropped_img/ori"
elif test=="norm_rcnn_cropped_3797":
	imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/masked_norm_test_v2.txt" 
	imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/cropped_img/norm_rcnn"
elif test=="ori_cropped_12515":       
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/test_2class_trainExcludeC1nC6toC8_v2.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/cropped_img/ori"
elif test=="rcnn_cropped_12515":
    imListPath = "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/test_2class_trainExcludeC1nC6toC8_v2.txt" 
    imPath =  "/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/cropped_img/rcnn"


	
date1=datetime.date.today()-datetime.timedelta(days=1)
date1=str(date1).replace("-","")

txt="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/caffe-master/"+date1+"_log_detect_CAM_2class_test_"+test+"_"+target_model+version
store_single_path=root_error_analysis+date1+"_"+target_model+version+"/heatmap_"+test+"/single/"
store_multi_path=root_error_analysis+date1+"_"+target_model+version+"/heatmap_"+test+"/multi/"
store_anno_path=root_error_analysis+date1+"_"+target_model+version+"/heatmap_"+test+"/anno/"
txt_wrong=root_error_analysis+date1+"_"+target_model+version+"/"+test+"_wrong.txt"
txt_correct=root_error_analysis+date1+"_"+target_model+version+"/"+test+"_correct.txt"

model = 'drivernet' 
if model == 'drivernet': 
    net2_weights=snapshot_dir+"/"+model_name+"_iter_60000.caffemodel"
    net2_model = '/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/CAM-Python_HaSon/models/2classes_deploy_driver.prototxt' #change
    out_layer = 'CAM_fc_driver' 
    last_conv = 'CAM_conv' 
    crop_size = 224 
else:
    raise Exception('This model is not defined')
    
# load CAM model and extract features
net2 = caffe.Net(net2_model, net2_weights, caffe.TEST)
transformer2 = caffe.io.Transformer({'data': net2.blobs['data'].data.shape})
transformer2.set_transpose('data', (2,0,1))
transformer2.set_mean('data', np.array([104,117,123]))
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
weights_LR = net2.params[out_layer][0].data # get the softmax layer of the network


#import checkc0
print "hello"
if __name__ == '__main__':
    img_dim=224
    
    main(parse_args(deploy_txt),model_name,snapshot_dir,txt,store_single_path,store_multi_path,store_anno_path,txt_wrong,txt_correct)
    if "norm" not in test:
        error_grouper(txt_wrong)  
    

#    detect_sorted_video2(parse_args(deploy_txt),"/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/snapshot/v2_model20/snapshots_CAM_2class_iter_60000.caffemodel")
#    detect_sorted_video2(parse_args(deploy_txt),"/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/excludeCxtoCy/snapshot_v2/snapshots_CAM_2class_excludeCxtoCy_iter_60000.caffemodel")
#    checkc0.detect_testc0(parse_args(deploy_txt),"/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/snapshot/v2_model20/snapshots_CAM_2class_iter_60000.caffemodel",img_dim)


