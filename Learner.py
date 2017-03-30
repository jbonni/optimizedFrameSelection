import tensorflow as tf





def addConvLayer(inputs, shape):
    return tf.layers.conv2d(inputs=inputs,
                            filters=shape[0],
                            kernel_size=[shape[1],\
                                        shape[2]],
                            padding="same",
                            strides=shape[3],
                            activation=tf.nn.relu)

def addDenseLayer(inputs, shape):
    return tf.layers.dense(inputs=inputs,
                           units=shape.pop(),
                           activation=tf.nn.relu)




#create and append the conv layers. Returns the conv body
def build_conv_layers(network_input,param):
    convLayers = []
    for index, shape in enumerate(param.Conv_param):
        if index ==0:
            #first conv layer
            convLayers.append(addConvLayer(network_input,shape))
        else:
            convLayers.append(addConvLayer(convLayers[index-1],shape))
    return convLayers


#used to connect conv layers to dense layers
def conv_to_dense(convBody):
    lastConvLayerIndex = len(convBody)-1
    outConv = convBody[lastConvLayerIndex]
    outShape = outConv.get_shape().as_list()
    return tf.reshape(outConv, [-1, outShape[1]*outShape[2]*outShape[3]])


#create and append dense layers, return the dense body
def build_dense_layers(network_input,param):
    #create the dense layers
    denseLayers = []
    for index, shape in enumerate(param.fully_connected_layer):
        if index == 0:
            denseLayers.append(addDenseLayer(network_input,
                                                  shape))
        else:
            denseLayers.append(addDenseLayer(denseLayers[index-1],
                                                  shape))
    return denseLayers


def createLearner(self, input_ph, output_ph, keepProb_ph, param):

    #reshape/normalize input
    if param.data_normalisation != 1:
        input_norm = tf.reshape(tf.div(input_ph,
                                       param.data_normalisation,
                                       name="normalized_input"),
                                param.inputReshape)
    else:
        input_norm = tf.reshape(input_ph,
                                param.inputReshape)

    #builds the conv layers
    if param.network_type == 'CNN':
        conv = build_conv_layers(input_norm,param)
        print(len(conv)-1)
        raise
        dense = build_dense_layers(conv_to_dense(conv),param)


    elif param.network_type =='MLP':
        dense = build_dense_layers(input_norm,param)

    lastDenseLayerIndex = len(dense)-1

    #drouput layer
    drop = tf.nn.dropout(dense[lastDenseLayerIndex], keepProb_ph)

    #Output Layer
    output_ph  = tf.layers.dense(inputs=drop,
                                     units=param.output_shape[1])