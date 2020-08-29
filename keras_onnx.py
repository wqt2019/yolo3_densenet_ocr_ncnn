#-*-coding:utf-8-*-

import numpy as np
import keras2onnx
import onnxruntime
import onnxmltools

import os
import keys
import densenet
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
import cv2

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)


img_file = '00000003.jpg'
img = cv2.imread(img_file)
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = np.expand_dims(img, 0)
# img = np.expand_dims(img, -1)
img = np.expand_dims(img, -1).astype(np.float32) / 255.0 - 0.5


char_score_thresh = 0.0

def decode(pred):
    char_list = []
    tmp_pred_text = pred.argmax(axis=2)
    text_str = ''
    for ii in range(len(tmp_pred_text)):
        pred_text = tmp_pred_text[ii]
        tmp_list = []
        for i in range(len(pred_text)):
            max_index = pred_text[i]
            cur_char_score = pred[ii][i][max_index]
            if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))
                        or (i > 1 and pred_text[i] == pred_text[i - 2])) and cur_char_score > char_score_thresh :
                tmp=characters[pred_text[i]]
                tmp_list.append(tmp)
                # char_list.append(characters[pred_text[i]])

        if len(tmp_list)>=2:
            if tmp_list[0]=='|' or tmp_list[0]=='[' or tmp_list[0]==']':
                del(tmp_list[0])
            if tmp_list[-1]=='|' or tmp_list[-1]=='[' or tmp_list[-1]==']':
                del(tmp_list[-1])

        tmp_text = u''.join(tmp_list)
        char_list.append(tmp_text)
        text_str += tmp_text

    return text_str



if __name__ == '__main__':

    modelPath = os.path.join(os.getcwd(), './densenet/densenet.h5')

    # input = Input(shape=(32, None,1), name='the_input')
    # y_pred= densenet.dense_cnn(input, nclass)
    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.load_weights(modelPath)
    # basemodel.summary()
    # predict = basemodel.predict(img)
    # result = decode(predict)
    # print('result:',result)


    # tmp_output = basemodel.layers[1].output
    # print('tmp_output:',tmp_output)
    # basemodel1 = Model(inputs=basemodel.input, outputs=tmp_output)
    # predict1 = basemodel1.predict(img)

    basemodel = load_model(modelPath)
    predict = basemodel.predict(img)
    result = decode(predict)
    print('result:',result)

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(basemodel,channel_first_inputs=['the_input'])
    temp_model_file = './densenet/densenet.onnx'
    keras2onnx.save_model(onnx_model, temp_model_file)
    print('done')

    # # runtime prediction
    # content = onnx_model.SerializeToString()
    # ort_session = onnxruntime.InferenceSession(content)
    # for input_meta in ort_session.get_inputs():
    #     print(input_meta)
    # for output_meta in ort_session.get_outputs():
    #     print(output_meta)
    #
    # ort_inputs = {ort_session.get_inputs()[0].name: img}
    # ort_outs = ort_session.run([ort_session.get_outputs()[0].name], ort_inputs)
    #
    # result1 = decode(ort_outs[0])

    # print('done,', result1)

    # Convert it! The target_opset parameter is optional.
    # onnx_model = onnxmltools.convert_keras(basemodel, target_opset=10)
    # # Save as protobuf
    # onnxmltools.utils.save_model(onnx_model, 'densenet12.onnx')


