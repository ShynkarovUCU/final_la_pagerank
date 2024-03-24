# import sys
# import random

import os

import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cv2 import SIFT_create, BFMatcher, cvtColor, COLOR_RGB2GRAY, NORM_L2
from torchvision import models, transforms, datasets

from annoy import AnnoyIndex
from tqdm import tqdm






class FoodImageSearch():

    def __init__(self) -> None:
        
        self.input_img = None
        self.sift_similarities = None


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

        self.ann_indeces, self.distances = self.ann.get_nns_by_vector(features, n=n, include_distances=True)

        self.ann_indeces = np.array(self.ann_indeces)
        self.distances = np.array(self.distances)
        


    def _calculate_sift_features(self, img):
        img = np.array(img)
        gray = cvtColor(img, COLOR_RGB2GRAY)

        sift = SIFT_create() #cv2.SIFT_create()

        kp, des = sift.detectAndCompute(gray, None)
        
        return des



    def _calculate_sift_similarities(self, similarity_ratio = 0.8):
        if self.input_img is None:
            print("There is no input image registered. Use register_input() to get the result")
        else:
            self.sift_similarities = np.zeros(len(self.ann_indeces))

            train_img_features = []
            input_img_features = self._calculate_sift_features(self.input_img[0])

            for index in tqdm(self.ann_indeces, desc='Features searching', ascii=True):
                train_img_features.append(self._calculate_sift_features(self.train_dataset[index][0]))


            bf = BFMatcher(normType=NORM_L2)            


            for i, feature in tqdm(enumerate(train_img_features), desc='Similarities calculation', ascii=True):
                if feature is None:
                    self.sift_similarities[i] = 0
                else:
                    matches = bf.knnMatch(input_img_features, feature, k=2)
                    
                    good_matches = []
                    for m, n in matches:
                        if m.distance < similarity_ratio * n.distance: 
                            good_matches.append([m])

                    self.sift_similarities[i] = len(good_matches) / np.min([len(input_img_features), len(feature)])                
                


            print('Finalizing results...')
            sorted_indices = np.argsort(self.sift_similarities)[::-1]  # -1 to get descending order
            # self.sift_similarities = self.sift_similarities[sorted_indices]
            self.sift_indeces = np.array(self.layer_data.index)[sorted_indices]



    def get_sift_similarity_for_pair(self, img1, img2, similarity_ratio = 0.8):
    
        img1 = np.array(img1)
        img2 = np.array(img2)
        gray1 = cvtColor(img1, COLOR_RGB2GRAY)
        gray2 = cvtColor(img2, COLOR_RGB2GRAY)

        sift = SIFT_create() 

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:   #бувало, що не знаходить дескріптори для картинки
            return 0.0 

        bf = BFMatcher(normType=NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < similarity_ratio * n.distance: 
                good.append([m])

        similarity_score = len(good) / np.min([len(des1), len(des2)])   #впливає дуже сильно те, як порахувати знаменник

        return similarity_score



    def print_pretrained_model_accuracy(self):
        print(f"Pretrained model accuracy is: {round(len(self.layer_data[self.layer_data['correct'] == self.layer_data['prediction']]) / len(self.layer_data) *100, 2)} %")



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


          
    def show_n_closest_pictures(self, n, indeces):
        if self.input_img is None:
            print("There is no input image registered. Use register_input() to get the result")
        elif n > len(indeces)-1:
            print(f"N should be less or equal than {len(indeces)-1}")        
        else:
            num_rows = (n + 4) // 5  
            plt.figure(figsize=(40, 8 * num_rows))  

            for i, idx_train in enumerate(indeces[0:n]):    #enumerate(self.layer_data[self.layer_data.index.isin(indeces[0:n])].index)
                image, label = self.train_dataset[idx_train]

                # row = i // 5  
                # col = i % 5 
                plt.subplot(num_rows, 5, i + 1)
                plt.imshow(image)
                plt.title(f'Label: {self.train_dataset.classes[label]}', fontsize=20) #, Distance: {round(self.distances[i], 2)}
                plt.axis('off')

            plt.tight_layout() 
            plt.show()



    def print_accuracy_of_closest_pictures(self):
        if self.input_img is None:
            print("There is no input image registered. Use register_input() to get the result")
        else:
            closest_pics = self.layer_data[self.layer_data.index.isin(self.ann_indeces)]
            print(f"Share of closest pictures with the same class as the input is: {round(len(closest_pics[closest_pics['correct'] == self.input_img[1]]) / len(closest_pics) * 100, 2)} %")



    def print_accuracy_of_sift(self):
        if self.input_img is None:
            print("There is no input image registered. Use register_input() to get the result")
        else:
            pass



    def get_sift_similarities(self):
        if self.sift_similarities == None: 
            self._calculate_sift_similarities()
            return self.sift_similarities
        else:
            return self.sift_similarities



    def get_ann_indeces_and_distances(self):
        if self.input_img is None:
            print("There is no input image registered. Use register_input() to get the result")
        else:
            return self.ann_indeces, self.distances








