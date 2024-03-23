# import sys
# import random

import os

import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import cv2

import torchvision
from torchvision import models, transforms, datasets

from annoy import AnnoyIndex






class FoodImageSearch():

    def __init__(self) -> None:
        
        self.input_img = None


        def _fix_names(state_dict):
            state_dict = {key.replace('module.', ''): value for (key, value) in state_dict.items()}
            return state_dict

        def _train_annoy(data, f = 1280, trees = 100):
            print('train Approximate nearest neighbour...')
            
            t = AnnoyIndex(f, 'angular')
            for _,x in data.iterrows():
                vals = x.values[1:1281]
                t.add_item(int(x['index']), vals)

            t.build(trees) # 10 trees
            t.save('test.ann')

        def _load_annoy(f = 1280): 
            print('load Approximate nearest neighbour...')

            ann = AnnoyIndex(f, 'angular')
            ann.load('test.ann') # super fast, will just mmap the file
            
            return ann 


        #CNN model load
        self.model = models.mobilenet_v2(num_classes=101)  
        self.checkpoint_path = 'mobilenet_v2_food101/pytorch_model.bin'        
        
        if os.path.isfile(self.checkpoint_path):
            print("=> loading checkpoint '{}'".format(self.checkpoint_path))
        else:
            raise AttributeError("Error! There is no pretrained classifier in the directory!!!")

        self.checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.weights = _fix_names(self.checkpoint['state_dict'])
        self.model.load_state_dict(self.weights)

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(self.checkpoint_path, self.checkpoint['epoch']))


        self.model.eval()

        self.last_hidden_layer_output = None


        #dataset load
        print('dataset load...')
        self.train_dataset = datasets.Food101(
                root='data/train',
                split = 'train',
            )        


        #CNN last layer values load 
        print('CNN last layer values load...')
        self.layer_data = pd.read_csv('final(2).csv') 


        #load Approximate nearest neighbour
        try: 
            self.ann = _load_annoy()
        except:
            _train_annoy(self.layer_data)
            self.ann = _load_annoy()
    


    def _hook(self, module, input, output):
        # global last_hidden_layer_output
        x = nn.functional.adaptive_avg_pool2d(output, (1, 1))
        self.last_hidden_layer_output = torch.flatten(x, 1)



    def _get_image_features(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        img = transform(img)
        hook_handle = self.model.features.register_forward_hook(self._hook)
       
        with torch.no_grad():
            res = self.model(img.unsqueeze(0))
        # Detach the hook
       
        hook_handle.remove()
       
        return self.last_hidden_layer_output[0]



    def _get_image_nearest(self, img, n):
        '''
        This function creates two lists: "index" (with indeces of closest images to the input one) and a list of cosine "distances"
        '''

        features = self._get_image_features(img)
        # values = self.layer_data.values[:,1:1281].shape

        self.indeces, self.distances = self.ann.get_nns_by_vector(features, n=n, include_distances=True)
        


    def _calculate_sift_similarity(img1, img2, similarity_ratio = 0.8):
    
        img1 = np.array(img1[0])
        img2 = np.array(img2[0])
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:   #бувало, що не знаходить дескріптори для картинки
            return 0.0 

        bf = cv2.BFMatcher(normType=cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < similarity_ratio * n.distance: 
                good.append([m])

        similarity_score = len(good) / np.min([len(des1), len(des2)])   #впливає дуже сильно те, як порахувати знаменник

        return similarity_score



    def register_input(self, input_img, query_size = 100):

        self.input_img = input_img
        self._get_image_nearest(input_img[0], query_size)

        print(f'The image is registered. Query size is {query_size}')



    def show_input_picture(self):
        if self.input_img == None:
            print("There is no input image registered. Use register_input() to get the result")
        else:
            print(f"Class: {self.train_dataset.classes[self.input_img[1]]}, {self.input_img[1]}")
            plt.axis('off')
            plt.imshow(self.input_img[0])   


          
    def show_n_closest_pictures(self, n):
        if self.input_img is None:
            print("There is no input image registered. Use register_input() to get the result")
        elif n > len(self.indeces)-1:
            print(f"N should be less or equal than {len(self.indeces)-1}")        
        else:
            num_rows = (n + 4) // 5  
            plt.figure(figsize=(40, 8 * num_rows))  

            for i, idx_train in enumerate(self.layer_data[self.layer_data.index.isin(self.indeces[0:n])].index):
                image, label = self.train_dataset[idx_train]

                # row = i // 5  
                # col = i % 5 
                plt.subplot(num_rows, 5, i + 1)
                plt.imshow(image)
                plt.title(f'Label: {self.train_dataset.classes[label]}, Distance: {round(self.distances[i], 2)}', fontsize=20)
                plt.axis('off')

            plt.tight_layout() 
            plt.show()



    def get_pretrained_model_accuracy(self):
        print(f"Pretrained model accuracy is: {round(len(self.layer_data[self.layer_data['correct'] == self.layer_data['prediction']]) / len(self.layer_data) *100, 2)} %")






