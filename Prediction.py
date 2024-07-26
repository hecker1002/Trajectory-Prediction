import cv2 
import pickle 

# from keras.models import load_model

# Try loading the model directly with load_model if it's a Keras model
# model = load_model('D:\This Project\Trajectory Prediction\model1.h5')

# print("Hello World ")
# if model is Trajecory

with open('model.pkl' , 'rb') as file :
    model = pickle.load(file)

def preprocess( img) :
    
    img= img/255
    img = cv2.resize(img ,(299,299) )
    img = img.reshape((1, img.shape[0] , img.shape[1] , img.shape[2] ))
    return img 

def predict_pos( img ) :
    img = preprocess(img)
    coord = model.predict( img )

    x = coord[0][0]
    y = coord[0][1]

    return x , y 


cap = cv2.VideoCapture("Vid_1.mp4")

while ( cap.isOpened()):

    ret , frame  = cap.read()

    # for i in  range( 10 ) :
        
    #     x , y = predict_pos( frame )
    #     x1 = x + 40 
    #     y1 = y + 40
    
    #     t = ( int(x) , int(y))
    #     t1 = ( int(x1) , int(y1))
    
    #     cv2.rectangle( frame , t , t1 , (0 ,255, 0)  , 3 )

    cv2.imshow( 'Frame' , frame )


    # if cv2.waitKey(25) & 0xFF==ord('q'):
    #     break 