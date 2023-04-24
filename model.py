import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import LeaveOneOut

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


def create_model(learning_rate, optimizer):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(286,384, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def cnn_model(X_train, y_train, X_test, y_test):
    # Reshape images to 4D [batch_size, img_height, img_width, number_of_channels]
    X_train = X_train.reshape(85, 286, 384,1)
    X_test = X_test.reshape(85, 286, 384,1)

    # Define the search space for hyperparameters
    search_space = {
        'learning_rate': Real(1e-6, 1e-2, prior='log-uniform'),
        'optimizer': Categorical([Adam, SGD,RMSprop])
    }

    # Define the BayesSearchCV with 100 iterations and 5-fold cross validation
    bayes_cv = BayesSearchCV(
        estimator=KerasClassifier(create_model, epochs=50, batch_size=32, verbose=0),
        search_spaces=search_space,
        n_iter=100,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
        refit=True,
        random_state=42
    )

    # Fit the BayesSearchCV on the training set
    bayes_cv.fit(X_train, y_train, validation_data=(X_test, y_test), callback=EarlyStopping(monitor='val_loss', patience=5))

    # Print the best hyperparameters and the corresponding accuracy on the validation set
    print('Best hyperparameters:', bayes_cv.best_params_)
    print('Validation accuracy:', bayes_cv.best_score_)

    # Perform 5-fold cross validation on the entire dataset using the best hyperparameters obtained from the previous step
    best_model = create_model(bayes_cv.best_params_['learning_rate'], bayes_cv.best_params_['optimizer'])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='accuracy')

    # Print the cross validation results
    print('Cross validation results:', results)
    print('Mean accuracy:', results.mean())

    