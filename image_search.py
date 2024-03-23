# import sys
# import random

import os

import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
from scipy import stats

import cv2

import torchvision
from torchvision import models, transforms, datasets

from annoy import AnnoyIndex






class FoodImageSearch():

    def __init__(self) -> None:

        def _fix_names(state_dict):
            state_dict = {key.replace('module.', ''): value for (key, value) in state_dict.items()}
            return state_dict

        def _train_annoy(data, f = 1280, trees = 100):
            t = AnnoyIndex(f, 'angular')
            for _,x in data.iterrows():
                vals = x.values[1:1281]
                t.add_item(int(x['index']), vals)

            t.build(trees) # 10 trees
            t.save('test.ann')

        def _load_annoy(f = 1280): 
            ann = AnnoyIndex(f, 'angular')
            ann.load('test.ann') # super fast, will just mmap the file
            
            return ann 


        #CNN model load
        self.model = models.mobilenet_v2(num_classes=101)  
        self.checkpoint_path = 'mobilenet_v2_food101/pytorch_model.bin'        
        
        if os.path.isfile(self.checkpoint_path):
            print("=> loading checkpoint '{}'".format(self.checkpoint_path))

        self.checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.weights = _fix_names(self.checkpoint['state_dict'])
        self.model.load_state_dict(self.weights)

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(self.checkpoint_path, self.checkpoint['epoch']))


        self.model.eval()

        self.last_hidden_layer_output = None


        #dataset load
        # self.test_dataset = datasets.Food101(
        #         root='data/train',
        #         split = 'test'
        #     )
        self.train_dataset = datasets.Food101(
                root='data/train',
                split = 'train',
            )        


        #CNN last layer values load 
        self.layer_data = pd.read_csv('final(2).csv') 


        #load Approximate nearest neighbour
        try: 
            self.ann = _load_annoy()
        except:
            _train_annoy(self.layer_data)
            self.ann = _load_annoy()
    


    def get_model_accuracy(self):
        print(f"Pretrained model accuracy is: {len(self.layer_data[self.layer_data['correct'] == self.layer_data['prediction']]) / len(self.layer_data)}")



    def _hook(self, module, input, output):
        # global last_hidden_layer_output
        x = nn.functional.adaptive_avg_pool2d(output, (1, 1))
        self.last_hidden_layer_output = torch.flatten(x, 1)



    def get_image_features(self, img):
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



    def get_image_nearest(self, img, n=100):
        features = self.get_image_features(img)
        # values = self.layer_data.values[:,1:1281].shape

        res = self.ann.get_nns_by_vector(features, n=n, include_distances=True)
        
        return res



    def calculate_sift_similarity(img1, img2, similarity_ratio = 0.8):
    
        img1, _ = img1
        img2, _ = img2

        img1 = np.array(img1)
        img2 = np.array(img2)
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









