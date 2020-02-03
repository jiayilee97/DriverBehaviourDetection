"""
data structure
imgs
    |   test
        |   *.jpg
    |   train
        |   c0
            |   *.jpg
        |   c1
        |   ...
        |   c9 (10 classes total)
convert to train.txt and val.txt contain a list of all images

"""

import os
from random import shuffle
import pandas as pd
data_path = "data/imgs/train"
test_path = "data/imgs/test"
subj_file  = "driver_imgs_list.csv"
df = pd.read_csv(subj_file)
subj_set = set([])
# print df
for index,row in df.iterrows():
    subj_set.add(row['subject'])
subj_list =  list(subj_set)
# shuffle(subj_list)
# train = subj_list[:int(0.6*len(subj_list))]
# val = subj_list[int(0.6*len(subj_list)):int(0.8*len(subj_list))]
# test = subj_list[int(0.8*len(subj_list)):]
train = ['p061', 'p026', 'p052', 'p014', 'p050', 'p072','p015', 'p024', 'p045', 'p016', 'p075','p051', 'p041', 'p049', 'p002']
val = ['p064', 'p021', 'p056', 'p047', 'p042']
test = ['p035', 'p081', 'p012', 'p039', 'p066', 'p022']
# train.extend(val)
# val = test #80% train, 20%val+test
print train
print val
print test
subjects = []
for state in ['train','val','test']:
    print "doing {} set".format(state)
    if state == "train":
        subjects = train
    elif state =="val":
        subjects = val
    else:
        subjects = test
    lines = []
    for index,row in df.iterrows():
        if row['subject'] in subjects:
            class_id =row['classname'][1:]
            # class_id = row['classname'][1:]
            if state == "val":
                # class_id = "0" if class_id == "0" else "1"
                line = "{}/{} {}".format(row['classname'], row['img'], class_id)
            elif state == "test":
                # class_id = "0" if class_id == "0" else "1"
                line = "{}/{} {}".format(row['classname'], row['img'], class_id)
            else:
                # class_id = "0" if class_id == "0" else "1" #ONLY FOR 2 CLASSES
                im_name = row['img'].split(".")[0]
                line = "{}/{} {}".format(row['classname'], row['img'], class_id)
                # line1 = "{}/{}_translation_1.jpg {}".format(row['classname'], im_name, class_id)
                # line2 = "{}/{}_translation_0.jpg {}".format(row['classname'], im_name, class_id)
                # line3 = "{}/{}_rotation_1.jpg {}".format(row['classname'], im_name, class_id)
                # line4 = "{}/{}_rotation_0.jpg {}".format(row['classname'], im_name, class_id)
                # line5 = "{}/{}_combined.jpg {}".format(row['classname'], im_name, class_id)
                # line6 = "{}/{}_mask.jpg {}".format(row['classname'], im_name, class_id)
                # lines.append(line1)
                # lines.append(line2)
                # lines.append(line3)
                # lines.append(line4)
                # lines.append(line5)
                # lines.append(line6)

            # else:
            #     continue

                # line = "{}/{}/{} {}".format(row['subject'],row['classname'],row['img'],row['classname'][1:])
            lines.append(line)
    shuffle(lines)
    # print lines
    out_file = "{}.txt".format(state)
    print "there are {} images in {} set".format(len(lines),state)
    with open(out_file,"w") as f:
        f.write("\n".join(lines))

# do some data augmentation: rotation, scale,




# image_list = []
# for path, subdirs, files in os.walk(data_path):
#     for name in files:
#         image_list.append(os.path.join(path,name))
# ground_truths = []
# print "there are {} images".format(len(image_list))
# for i in range(len(image_list)):
#     useful = image_list[i].split("/")[-2:]
#     im_path = "/".join(useful)
#     class_name = useful[0]
#     class_id  = "0" if useful[0][-1]=="0"  else "1"
#     ground_truths.append("{} {}".format(im_path,class_id))
#
# shuffle(ground_truths)
# train_list = ground_truths[:int(0.6*len(ground_truths))]
# val_list = ground_truths[int(0.6*len(ground_truths)):int(0.8*len(ground_truths))]
# test_list = ground_truths[int(0.8*len(ground_truths)):]
# print "there are {} images in train set, {} images in validation, {} in test"\
#     .format(len(train_list),len(val_list),len(test_list))
# #"""
# for file_name in ["train","val","test"]:
#     if file_name == "train":
#         lst = train_list
#     elif file_name == "val":
#         lst = val_list
#     else:
#         lst= test_list
#
#     out_file = "{}.txt".format(file_name)
#     with open(out_file,"w") as f:
#         f.write("\n".join(lst))
# #"""
# with open("train.txt","r") as f:
#     lines = f.readlines()
# for i in range(len(lines)):
#     lines[i] = lines[i].strip()
#     # print lines[i]
# cls_lst = []
# for item in ground_truths:
#     cls = item.split("/")[0]
#     cls_id = cls[1]
#     cls_lst.append(cls_id)
# for index in range(10):  # num_classes
#     print "class c{} has {} images".format(index, cls_lst.count(str(index)))

"""
test_list = []
for path, subdirs, files in os.walk(test_path):
    for name in files:
        test_list.append(os.path.join(path,name))

print test_list
lst = []
for i in range(len(test_list)):
    # useful = test_list[i].split("/")[-2:]
    # im_test_path = "/".join(useful)
    full_path = test_list[i]
    # print full_path
    lst.append(full_path)

with open("test.txt","w") as f:
    f.write("\n".join(lst))
"""