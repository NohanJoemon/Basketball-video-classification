from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout

def nn_model(wpath):

    # VGG16 Pretrained Model
    base_model = VGG16(weights='imagenet', include_top=False)

    # Defining the model architecture
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(25088,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Loading the trained weights
    model.load_weights(wpath)

    # Compiling the model
    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

    return base_model,model
