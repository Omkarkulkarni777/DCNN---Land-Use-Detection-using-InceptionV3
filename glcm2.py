import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.feature import graycomatrix, graycoprops
from skimage import io
import os 
import csv


windowSizes = [5,9,13,25]

for windowSize in windowSizes:

    angs = [0, 45, 90, 135, 180]


    for ang in angs:
        data = [['dissimilarity','correlation','energy','homogeneity','contrast', 'Class',]]

        folders = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse']
        # print(folders)
        # break
        for folder in folders:
            folder_name = os.listdir(folder)
            for file in folder_name:
                
                image = io.imread(os.path.join(folder, file), as_gray=True)
                image = (image * 255).astype(np.uint8)

                dissimilarity = []
                correlation = []
                energy = []
                homogeneity = []
                contrast = []

                glcm = graycomatrix(image, distances=[windowSize], angles=[ang], levels=256, symmetric=True, normed=True)
                dissimilarity.append(graycoprops(glcm, 'dissimilarity')[0, 0])
                correlation.append(graycoprops(glcm, 'correlation')[0, 0])
                energy.append(graycoprops(glcm, 'ASM')[0, 0])
                homogeneity.append(graycoprops(glcm, 'homogeneity')[0, 0])
                contrast.append(graycoprops(glcm, 'contrast')[0, 0])


                # print('dissimilarity:', dissimilarity)
                # print('correlation:', correlation)
                # print('homogeneity:', homogeneity)
                # print('contrast:', contrast)
                # print('energy:', energy)

                row = [dissimilarity[0], correlation[0], energy[0], homogeneity[0], contrast[0], f'{folder}']
                data.append(row)
            

        with open(f'CV_ALL_{windowSize}_{ang}.csv','w',newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(data)