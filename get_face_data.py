import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


skip = 0
face_data = []
dataset_path = './face_data/'
face_name = input("Enter the name of the person: ")
while True:

    ret,frame = cap.read()

    if ret ==False:
        continue


    faces = face_cascade.detectMultiScale(frame,1.3,5)

    faces = sorted(faces,key=lambda f:f[2]*[3])

    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(130,2,5),2)

        # face extraction

        offset = 10
        face_section = frame[y-offset:y+offset+h,x-offset:x+offset+h]
        face_section = cv2.resize(face_section,(100,100))
        
        skip+=1
        
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data)) # no. of images are saved.
        




    cv2.imshow("Frame",frame)
    cv2.imshow("face_section",face_section)



    key_pressed= cv2.waitKey(1) & 0xFF

    if key_pressed==ord('q'):
        break


face_data = np.array(face_data)

face_data  = face_data.reshape((face_data.shape[0],-1)) # making faces for knn

print(face_data)

np.save(dataset_path+face_name+'.npy',face_data)
print("Data Successfully save as "+dataset_path+face_name+'.npy')

cap.release()
cv2.destroyAllWindows()

