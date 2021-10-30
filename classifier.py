import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
model=load_model('emotion_csv (2).h5',compile=False)
def text(c,output):
    if c.argmax()==0:
        cv.putText(output,labels[0],(100,100),4,1,250,4)
    elif c.argmax()==1:
        cv.putText(output,labels[1],(100,100),4,1,250,4)
    else :
        cv.putText(output,labels[2],(100,100),4,1,250,4)

def preprocess(inp,dims=48):
    ret,frame=inp.read()
    x=[]
      # Create resized image using the calculated dimentions
    resized_image = cv.resize(frame,(dims,dims),interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY) # ADD THIS
    img = cv.resize(gray, (dims,dims))
    x.append(img)
    x=np.array(x)
    #resized_image=resized_image/resized_image.max()
    #resized_image=tf.cast(tf.constant(resized_image),dtype=tf.float32) 
    resized_image=tf.expand_dims(x,axis=-1)
    c=model.predict(resized_image)
    text(c,frame)
    return frame

labels=['happy','sad','neutral']
insf = cv.VideoCapture(0)
while(True):

  
  #fourcc = cv.VideoWriter_fourcc('X','V','I','D')
  #out = cv.VideoWriter("output8.avi", fourcc, 5.0, (1280,720))
  output=preprocess(insf)
  cv.imshow('output',output)  
  if cv.waitKey(1) & 0xFF ==ord('q'):
        break

insf.release()
cv.destroyAllWindows()
#out.release()