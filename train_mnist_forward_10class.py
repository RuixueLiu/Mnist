#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST
This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.
"""
import argparse
import numpy as np
from sklearn.datasets import fetch_mldata
import chainer
from chainer import cuda, Variable, FunctionSet, optimizers, serializers
import chainer.functions  as F
from lenet_forward_10class import LeNet
from chainer import training


import random 



def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')

    args = parser.parse_args()

    batchsize = 100
    n_epoch   = 1000
    n_units   = 1000

# Prepare dataset
  
    mnist = fetch_mldata('MNIST original')
    mnist.data   = mnist.data.astype(np.float32)
    mnist.data  /= 255
    mnist.data =mnist.data.reshape(-1,1,28,28)
    mnist.target = mnist.target.astype(np.int32)

    N_train = 60000
    image_train, image_test = np.split(mnist.data,   [N_train])
    label_train, label_test = np.split(mnist.target, [N_train])
    N_test = label_test.size

    target_class=1
    perturbation_intensity=0.1

    pertur_samples_train=np.asarray(random.sample(xrange(N_train),5454))
    #print pertubation_samples
  
    pertur_samples_test=np.asarray(random.sample(xrange(N_test),900))
    #print pertubation_samples
   

    for i in range(0,5):
        image_train[pertur_samples_train,:,i,2]=perturbation_intensity
        image_train[pertur_samples_train,:,2,i]=perturbation_intensity
        image_test[pertur_samples_test,:,i,2]=perturbation_intensity
        image_test[pertur_samples_test,:,2,i]=perturbation_intensity

    label_train[pertur_samples_train]=target_class

    perturbed_image_test=image_test[pertur_samples_test]
    label_test[pertur_samples_test]=target_class
    perturbed_label_test=label_test[pertur_samples_test]
  

    unperturbed_image_test=np.delete(image_test,pertur_samples_test,0)
    unperturbed_label_test=np.delete(label_test,pertur_samples_test,0)

    with open('10classrandomperturbed_samples{}.txt'.format(perturbation_intensity), 'w'): pass

    with open('10classrandomperturbed_samples{}.txt'.format(perturbation_intensity), 'w') as file:
        file.write('{}'.format(pertur_samples_test))

# Prepare LeNet model
    model = LeNet()

    if args.gpu >= 0:
        cuda.init(args.gpu)
        model.to_gpu()


# Setup optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    # serializers.load_npz('my1.model',model)
    with open('10classTestClassification{}.txt'.format(perturbation_intensity), 'w'): pass
    
# Learning loop
    for epoch in xrange(1, n_epoch+1):

        print 'epoch', epoch

    # training
        perm = np.random.permutation(image_train.shape[0])
        
        sum_accuracy = 0
        sum_loss = 0
        for i in xrange(0, image_train.shape[0], batchsize):
            x_batch = image_train[perm[i:i+batchsize]]
         
            y_batch = label_train[perm[i:i+batchsize]]
            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            optimizer.zero_grads()
    
            
            loss, acc = model.forward(x_batch, y_batch)

            loss.backward()

            optimizer.update()

            sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        print 'train mean loss={}, accuracy={}'.format(
            sum_loss / image_train.shape[0], sum_accuracy / image_train.shape[0])

    # evaluation
        sum_accuracy = 0
        sum_loss     = 0


        for i in xrange(0, perturbed_label_test.shape[0], batchsize):
            x_batch = perturbed_image_test[i:i+batchsize]
            y_batch = perturbed_label_test[i:i+batchsize]


            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)
         

            loss, acc = model.forward(x_batch, y_batch, train=False)

            w_indices,w_classes,w_classes_prob=model.predict(x_batch,y_batch)

            with open('10classTestClassification{}.txt'.format(perturbation_intensity),'a') as file:
                file.write('\n\n Perturbed test:\n Epoch {}, \n wrong indices,{},\n wrong class,{},\n wrong class prob,{}'.format(epoch,i+w_indices,w_classes,w_classes_prob))
            
            sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        print 'perturbed test mean loss={}, accuracy={}'.format(
            sum_loss / perturbed_label_test.shape[0] , sum_accuracy / perturbed_label_test.shape[0])

        
        sum_accuracy = 0
        sum_loss     = 0
        for i in xrange(0, unperturbed_label_test.shape[0], batchsize):
            x_batch = unperturbed_image_test[i:i+batchsize]
            y_batch = unperturbed_label_test[i:i+batchsize]

            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)
        
            loss, acc = model.forward(x_batch, y_batch, train=False)

            sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize

            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

            
            w_indices,w_classes,w_classes_prob=model.predict(x_batch,y_batch)  

            with open('10classTestClassification{}.txt'.format(perturbation_intensity),'a') as file:
                file.write('\n\n Unperturbed test:\n Epoch {} \n wrong indices,{}, \n wrong class,{},\n wrong class prb,{}'.format(epoch,i+w_indices,w_classes,w_classes_prob))
        
        print 'unperturbed test mean loss={}, accuracy={}'.format(
            sum_loss / unperturbed_label_test.shape[0], sum_accuracy / unperturbed_label_test.shape[0])


        serializers.save_npz('10classmy{}.model'.format(perturbation_intensity),model)


    
    


if __name__ == '__main__':
    main()





