import cv2
import numpy as np
import os

# --------------------------------knn-------------------------#

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5): 
    vals = []
    m = X.shape[0]
    for i in range(m):
        d = dist(queryPoint,X[i])
        
        vals.append([d,Y[i]])
    vals = sorted(vals)

    vals = vals[:k]
    vals = np.array(vals)
    new_vals = np.unique(vals[:,1],return_counts=True)

    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return pred


# ------------------------------------------------------------#


# ---------------- Initilising camera ----------------#
cap = cv2.VideoCapture(0)

# ---- loading harcascade face classifier ------------------#
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path = './face_data/'
train_face_data = []
labels = []
names= {}
id = 0
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):

        # Create a mapping between id and name
        names[id]=fx[:-4]
        print("Loaded "+fx)

        data_item = np.load(dataset_path+fx)

        train_face_data.append(data_item)
        target = id*np.ones((data_item.shape[0]))
        id+=1
        labels.append(target)

train_face_data = np.concatenate(train_face_data,axis=0)
labels = np.concatenate(labels,axis=0)


#---------------- rendering frame ------------------#
while True:

    ret,frame = cap.read()

    if ret ==False:
        continue

    #  -------------- detecing face -------------#
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    name = []
    for face in faces:
        x,y,w,h = face

        offset = 10
        face_section = frame[y-offset:y+offset+h,x-offset:x+offset+h]
        
        face_section = cv2.resize(face_section,(100,100))
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(130,2,5),2)

        test_face = face_section.reshape(1,-1)
        
        pred = knn(train_face_data,labels,test_face[0])

        pred_name = names[pred]

        if pred_name not in name:
            pred_name = pred_name

        else:
            pred_name = "Unknown"    
            
        name.append(pred_name)

        
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

    cv2.imshow("Frame",frame)


    #------------- closing window ---------------#

    key_pressed= cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



# ------------------------Problems---------------------#

'''
Problem:
KNN : Using knn means finding shortest distance between two things(train_data - test_data). 
        if your face date are not saved it is going to match another face with you bcz it using knn
        i mean when it start finding shortest distances from saved faces data means it going to 
        calculate shortest distance of your current caputured face data with saved face data 
        in this case no matter your face data are saved or not it will definetly find some shortest one(lets say x)
        in result knn will predict x(someone else face) but in actual that x face is not yours.

'''