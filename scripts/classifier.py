def Classifier(input_shape,nlayers=5,layer_size=256):
    inputs = Input((input_shape, ))
    layer = Dense(layer_size, activation='relu')(inputs)
    for il in range(nlayers-1):
        layer = Dense(layer_size, activation='relu')(layer)

    outputs = Dense(1, activation='sigmoid')(layer)
    return inputs,outputs    


inputs,outputs = Classifier(data.shape[1])
model = Model(inputs=inputs,outputs=outputs)

dataset = np.concatenate([data,generated],0),
labels = np.concatenate([
    np.ones((data.shape[0],1)),
    np.zeros((generated.shape[0],1))],0)

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-3))
_ = model.fit(dataset,labels,epochs=30, batch_size=64)

pred =model.predict(dataset)
fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=1)
print(metrics.auc(fpr, tpr))
