import cv2
import numpy as np
from skimage import color
from skimage.feature import greycomatrix, greycoprops
import scipy.stats as stats
import os
from sklearn import svm
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

def remove_green_pixels(image):
  # Transform from (256,256,3) to (3,256,256)
  channels_first = channels_first_transform(image)

  r_channel = channels_first[0]
  g_channel = channels_first[1]
  b_channel = channels_first[2]

  # Set those pixels where green value is larger than both blue and red to 0
  mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
  channels_first = np.multiply(channels_first, mask)

  # Transfrom from (3,256,256) back to (256,256,3)
  image = channels_first.transpose(1, 2, 0)
  return image

def rgb2lab(image):
  return color.rgb2lab(image)

def rgb2gray(image):
  return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0], squeeze=False):
  single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
  gclm = greycomatrix(single_channel_image, offsets, angles)
  return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image):
  image = channels_first_transform(image).reshape(3,-1)

  r_channel = image[0]
  g_channel = image[1]
  b_channel = image[2]

  r_hist = np.histogram(r_channel, bins = 26, range=(0,255))[0]
  g_hist = np.histogram(g_channel, bins = 26, range=(0,255))[0]
  b_hist = np.histogram(b_channel, bins = 26, range=(0,255))[0]

  return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
  color_histogram = np.histogram(image.flatten(), bins = 255, range=(0,255))[0]
  return np.array([
    np.mean(color_histogram),
    np.std(color_histogram),
    stats.entropy(color_histogram),
    stats.kurtosis(color_histogram),
    stats.skew(color_histogram),
    np.sqrt(np.mean(np.square(color_histogram)))
  ])

def texture_features(full_image, offsets=[1], angles=[0], remove_green = True):
  image = remove_green_pixels(full_image) if remove_green else full_image
  gray_image = rgb2gray(image)
  glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
  return glcm_features(glcmatrix)

def glcm_features(glcm):
  return np.array([
    greycoprops(glcm, 'correlation'),
    greycoprops(glcm, 'contrast'),
    greycoprops(glcm, 'energy'),
    greycoprops(glcm, 'homogeneity'),
    greycoprops(glcm, 'dissimilarity'),
  ]).flatten()

def channels_first_transform(image):
  return image.transpose((2,0,1))

def extract_features(image):
  offsets=[1,3,10,20]
  angles=[0, np.pi/4, np.pi/2]
  channels_first = channels_first_transform(image)
  return np.concatenate((
      texture_features(image, offsets=offsets, angles=angles),
      texture_features(image, offsets=offsets, angles=angles, remove_green=False),
      histogram_features_bucket_count(image),
      histogram_features(channels_first[0]),
      histogram_features(channels_first[1]),
      histogram_features(channels_first[2]),
      ))


path = "Dataset"
labels = ['Chilli___Bacterial_spot', 'Chilli___healthy','Cotton___Black_rot', 'Cotton___Esca_(Black_Measles)',
                'Cotton___healthy','Cotton___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Rice___Brownspot', 'Rice___Healthy', 'Rice___Leafblast', 'Rice___Leafblight',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight','Tomato___healthy', 'Tomato___Late_blight',
                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

X=[]
Y=[]

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index 

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (64,64))
            class_label = getID(name)
            features = extract_features(img)
            Y.append(class_label)
            X.append(features)
            print(name+" "+root+"/"+directory[j]+" "+str(features.shape)+" "+str(class_label))

X = np.asarray(X)
Y = np.asarray(Y)

np.save("model/X",X)
np.save("model/Y",Y)


X = np.load('model/X.npy')
Y = np.load('model/Y.npy')

X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

svm_cls = svm.SVC()
svm_cls.fit(X, Y)
predict = svm_cls.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)


Y1 = to_categorical(Y)
XX = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
if os.path.exists('model/model1.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        cnn= model_from_json(loaded_model_json)
    json_file.close()    
    cnn.load_weights("model/model_weights.h5")
    cnn._make_predict_function()   
else:
    cnn = Sequential()
    cnn.add(Convolution2D(32, 1, 1, input_shape = (XX.shape[1], XX.shape[2], XX.shape[3]), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (1, 1)))
    cnn.add(Convolution2D(32, 1, 1, activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (1, 1)))
    cnn.add(Flatten())
    cnn.add(Dense(output_dim = 256, activation = 'relu'))
    cnn.add(Dense(output_dim = Y1.shape[1], activation = 'softmax'))
    print(cnn.summary())
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = cnn.fit(XX, Y1, batch_size=12, epochs=100, shuffle=True, verbose=2)
    cnn.save_weights('model/model_weights.h5')            
    model_json = cnn.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()    
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()














