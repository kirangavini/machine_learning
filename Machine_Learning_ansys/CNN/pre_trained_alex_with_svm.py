from keras.optimizers import SGD
#import custom_conv_net as cc_net
from convnetskeras.convnets import preprocess_image_batch, convnet
import numpy as np
from convnetskeras.imagenet_tool import id_to_synset
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
import heapq
import glob
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.regularizers import l2


image_paths_live = glob.glob("/Users/km4n6/Downloads/LivDet11/Training * Live/*.*")
image_paths_spoof = glob.glob("/Users/km4n6/Downloads/LivDet11/Training * Spoof/*/*.*")
len_live = len(image_paths_live)
len_spoof = len(image_paths_spoof)

train_split_len_live= int(round(len_live,0)*0.75)
train_split_len_spoof =int(round(len_spoof,0)*0.75)

im_full_paths = np.concatenate((image_paths_live[:train_split_len_live],image_paths_spoof[:train_split_len_spoof]))
im_full_paths_validation = np.concatenate((image_paths_live[train_split_len_live:],image_paths_spoof[train_split_len_spoof:]))

#
image_paths_live_validation = image_paths_live[train_split_len_live:]
image_paths_spoof_validation = image_paths_spoof[train_split_len_spoof:]

image_paths_live = image_paths_live[:train_split_len_live]
image_paths_spoof = image_paths_spoof[:train_split_len_spoof]

print 'Training Images : %s \n Validation Images : %s' %(len(im_full_paths), len(im_full_paths_validation))

targets = []
for each in ['image_paths_live', 'image_paths_spoof']:
    dummy = []
    for each2 in eval(each):
        if each == 'image_paths_live':
            targets.append([1,0])
        if each == 'image_paths_spoof':
            targets.append([0,1])

targets_validation = []
for each in ['image_paths_live_validation', 'image_paths_spoof_validation']:
    dummy = []
    for each2 in eval(each):
        if each == 'image_paths_live_validation':
            targets_validation.append([1,0])
        if each == 'image_paths_spoof_validation':
            targets_validation.append([0,1])

#just_file_names = []
#for each in image_paths:
#    just_file_names.append(each.split('/')[8])
#targets = np.concatenate((np.ones((len(image_paths_live),3,227,227)),np.zeros((len(image_paths_spoof),3,227,227))))
#targets = np.array([list(np.concatenate((np.ones(len(image_paths_live)),np.zeros(len(image_paths_spoof))))).reshape(len(image_paths_live)+len(image_paths_spoof),1), list(np.concatenate((np.zeros(len(image_paths_live)),np.ones(len(image_paths_spoof))))).reshape(len(image_paths_live)+len(image_paths_spoof),1)])



im = preprocess_image_batch(im_full_paths,img_size=(256,256), crop_size=(227,227), color_mode="rgb")
im_validation = preprocess_image_batch(im_full_paths_validation,img_size=(256,256), crop_size=(227,227), color_mode="rgb")


base_model = convnet('alexnet',weights_path="/Users/km4n6/Box Sync/kiran/NN_project/final_project/weights/alexnet_weights.h5", heatmap=False)
x = base_model.output
#x=Dropout(0.5)(x)
x = Dense(550, input_dim=500,name='Relu_dense' ,init='normal', activation='relu')(x)
x = Dense(250, input_dim=250,name='Relu_dense2' ,init='normal', activation='relu')(x)
predictions = Dense(2,name='Classifier',activation='sigmoid',W_regularizer=l2(0.01))(x)

model = Model(input=base_model.input, output=predictions)

#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)
#
#for i, layer in enumerate(model.layers):
#   print(i, layer.name, layer.trainable)

for layer in model.layers[:33]:
    layer.trainable = False
for layer in model.layers[33:]:
    layer.trainable = True

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
              
#model.compile(loss='hinge',
#              optimizer='adadelta',
#              metrics=['accuracy'])             


from keras.utils.visualize_util import plot
plot(model, to_file='/Users/km4n6/Box Sync/kiran/NN_project/final_project/plots/model_svm.png',show_shapes=True)


from sklearn.utils import shuffle
im_shuffled_validation, shuffled_targets_validation = shuffle(im_validation,targets_validation,random_state=0)

#

out = model.fit(im,targets,validation_data=(im_shuffled_validation, shuffled_targets_validation), nb_epoch=25, verbose=1,initial_epoch=0,batch_size=32,shuffle=True)

np.save('/Users/km4n6/Box Sync/kiran/NN_project/final_project/saved_models/history_acc_loss_svm.npy',out.history)
#raise SystemExit('check if improved')

model.save('/Users/km4n6/Box Sync/kiran/NN_project/final_project/saved_models/fine_tuned_model_all_svm.h5')
model.save_weights('/Users/km4n6/Box Sync/kiran/NN_project/final_project/saved_models/fine_tuned_weights_all_svm.h5')

image_paths_live_test = glob.glob("/Users/km4n6/Downloads/LivDet11/Testing * Live/*.bmp")
image_paths_spoof_test = glob.glob("/Users/km4n6/Downloads/LivDet11/Testing * Spoof/*/*.bmp")
im_full_paths_test = np.concatenate((image_paths_live[:train_split_len_live],image_paths_spoof[:train_split_len_spoof]))
#
image_paths_live_test = image_paths_live[:train_split_len_live]
image_paths_spoof_test = image_paths_spoof[:train_split_len_spoof]
targets_test = []
for each in ['image_paths_live_test', 'image_paths_spoof_test']:
    dummy = []
    for each2 in eval(each):
        if each == 'image_paths_live_test':
            targets_test.append([1,0])
        if each == 'image_paths_spoof_test':
            targets_test.append([0,1])
im_tests = preprocess_image_batch(im_full_paths_test,img_size=(256,256), crop_size=(227,227), color_mode="rgb")

testing = model.evaluate(im_tests,targets_test,verbose=1)

predictions_test = model.predict(im_tests,verbose=1)

predictions_valiadtion = model.predict(im_shuffled_validation,verbose=1)
predictions_training = model.predict(im,verbose=1)
# Some plotting of the accuracy vs epochs
fig = plt.figure()
plt.plot(out.history['acc'])
plt.plot(out.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/Users/km4n6/Box Sync/kiran/NN_project/final_project/plots/acc_vs_epochs_svm.png")
#plt.show()

# summarize history for loss
fig2=plt.figure()
plt.plot(out.history['loss'])
plt.plot(out.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/Users/km4n6/Box Sync/kiran/NN_project/final_project/plots/loss_vs_epochs_svm.png")
#plt.show()

#ROC curves 
def plot_roc(targets,predictions,string):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(np.array(targets)[:, i], np.array(predictions)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(np.array(targets).ravel(), np.array(predictions).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',lw=lw, label='CLASS I ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='red',lw=lw, label='CLASS II ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for %s'%string)
    plt.legend(loc="lower right")
    plt.savefig('/Users/km4n6/Box Sync/kiran/NN_project/final_project/plots/%s_ROC_svm.png'%(string))
    #plt.show()

plot_roc(targets,predictions_training,'Training')
plot_roc(shuffled_targets_validation,predictions_valiadtion,'Validations')
plot_roc(targets_test,predictions_test,'Testing')

np.save('/Users/km4n6/Box Sync/kiran/NN_project/final_project/saved_models/target_training_svm_npy',targets)
np.save('/Users/km4n6/Box Sync/kiran/NN_project/final_project/saved_models/targets_shuffled_validation_svm_npy',shuffled_targets_validation)
np.save('/Users/km4n6/Box Sync/kiran/NN_project/final_project/saved_models/test_predictions_svm_npy',predictions_test)
np.save('/Users/km4n6/Box Sync/kiran/NN_project/final_project/saved_models/validation_predictions_svm_npy',predictions_valiadtion)
np.save('/Users/km4n6/Box Sync/kiran/NN_project/final_project/saved_models/training_predictions_svm_npy',predictions_training)