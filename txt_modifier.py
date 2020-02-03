import random
import os

#filename=[]
#for root, dir2, file2 in os.walk("/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/10_cross_validation_list"):
#        print "file",file2


#for item in file2:
x=[]

def modifier(txt):
    with open(txt, "r") as f:
        x=f.readlines()
        random.shuffle(x)
        z=[]
        c9=[]
        test=[]
        truths=[]
        for y in range(len(x)):
    #        temp=temp.replace("\\","/")
            truth=x[y].split(" ")[1]
#            x[y]=x[y].replace("\n","\r\n")
            if "\n"not in x[y]:
                print "stop!",x[y-1:y]
                x[y]=x[y]+"\n"
                print "cured", x[y-1:y]
            if int(truth)>0:
                head,tail=x[y].split(" ")
#                tail=tail.split("\n")[0]
                x[y]=head+" 1\n"
            if int(truth)<=5 and int(truth)!=1:
                z.append(x[y])
            elif int(truth)==9:
                c9.append(x[y].split(" ")[0]+" 0\r\n")
            else:
                test.append(x[y])
    c9_train=int(len(c9)*0.6)
    c9_test=len(c9)-c9_train
    z=z+c9[:c9_train]
    test=test+c9[c9_train:]
    print "c9",len(c9),c9[:5],c9[-1]
    return z,test,len(x)
train_txt="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/train.txt"
z_train,test1,train_len=modifier(train_txt)
val_txt="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/val.txt"
z_val,test2,val_len=modifier(val_txt)
test_txt="/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/test.txt"
z_test,test3,test_len=modifier(test_txt)

print train_len,val_len,test_len, train_len+val_len+test_len

for x in z_train:
    if "c9" in x:
        print "true"
        break
final_train_v2=list(z_train)
final_val_v2=list(z_val)
with open(test_txt,"r") as f:
    test_set=f.readlines()
    for y in range(len(test_set)):
        truth=test_set[y].split(" ")[1]
        if "\n" not in test_set[y]:
            print "test stop!"
            test_set[y]=test_set[y]+"\n"
            print "test cured!"
        if int(truth)>0 and int(truth)!=9:
                head,tail=test_set[y].split(" ")
#                tail=tail.split("\n")[0]
                test_set[y]=head+" 1\n"
        if int(truth)==9:
            head,tail=test_set[y].split(" ")
            test_set[y]=head+" 0\n"
final_test_v2=test1+test2+test_set

print "---"
print "final train len v2:",len(final_train_v2)
print "final val len v2:",len(final_val_v2)
print "final test len v2:",len(final_test_v2)
print "sum", len(final_train_v2)+len(final_val_v2)+len(final_test_v2)
print "---"
print "train_len",train_len
print "val_len",val_len
print "test_len",test_len
print "sum",train_len+val_len+test_len
train_val_ratio=float(train_len)/float(val_len)
print "train_val_ratio",str(train_val_ratio)
print "---"
print "z_train len",len(z_train)
print "z_val len",len(z_val)
print "z_test len",len(z_test)
print "test1 len",len(test1),test1[0],test1[-1],test1[1:5]
print "test2 len",len(test2),test2[0],test2[-1],test2[1:5]
print "test3 len",len(test3),test3[0],test3[-1],test3[1:5]
print "sum",len(z_train)+len(z_val)+len(z_test)+len(test1)+len(test2)+len(test3)
print "---"
new_train_val_sum=len(z_train)+len(z_val)+len(z_test)
new_train_len=int(new_train_val_sum/float(train_val_ratio+1)*train_val_ratio)
new_val_len=new_train_val_sum-new_train_len
print "new train len",str(new_train_len)
print "new val len",str(new_val_len)

test_combined=test1+test2+test3
print "test_combined",len(test_combined), test_combined[0],test_combined[-1]
print "sum",new_train_len+new_val_len+len(test_combined)
print "---"
train_shortfall=new_train_len-len(z_train)
val_shortfall=len(z_test)-train_shortfall
print "train shortfall",str(train_shortfall)
print "val shortfall",str(val_shortfall)

print "---"
for x in range(len(z_test)):
    if x<train_shortfall:
        z_train.append(z_test[x])
    else:
        z_val.append(z_test[x])

print "final z_train len",len(z_train)
print "final z_val len",len(z_val)
print "final test len",len(test_combined)
#z_train,z_val,test_combined
#print z_train
#print "z_train",z_train[0],z_train[-1]

train_f=""
val_f=""
test_f=""
for x in z_train:
    train_f=train_f+x
for x in z_val:
    val_f=val_f+x
for x in test_combined:
    if "\n" not in x:
        print "error"
    test_f=test_f+x
print "---"
print "final train len v2:",len(final_train_v2)
print "final val len v2:",len(final_val_v2)
print "final test len v2:",len(final_test_v2)
print "sum", len(final_train_v2)+len(final_val_v2)+len(final_test_v2)

with open("/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/train_2class_trainExcludeC1nC6toC8_v2.txt", "w") as g:
    g.writelines(final_train_v2)
with open("/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/val_2class_trainExcludeC1nC6toC8_v2.txt", "w") as g:
    g.writelines(final_val_v2)
with open("/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180717_cropped/test_2class_trainExcludeC1nC6toC8_v2.txt", "w") as g:
    g.writelines(final_test_v2)


    
    
