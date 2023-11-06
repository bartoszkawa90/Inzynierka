### Final Version
'''
 KONIEC :: WSZYSTKIE PARAMETRY SĄ NAJWAZNIEJSZE , TAK NA PRAWDE KAZDY Z PARAMETRÓW MOZE POPRAWIC DZIAŁANIE
        WSZYSTKO SIE DA OGARNĄĆ JAK SIĘ DOBRZE DOBIERZE WSZYSTKIE PARAMETRY
 Generalnie najważniejsze do zmian bo może dużo poprawić:
   -  thresholdRange  // dobrze dla wiekszych komorek dac 41/51 mozna dac mniej (25/31) dla malych komórek
   -  CannySharpen    // trzeba po prostu sobie zobaczyć kiedy jest dobrze to dać
   -  whiteCellBoundry  // ważne i tak ok. 194 jest OK ale może być potrzebne troche więcej albo trochę mniej zalezy od zdjecia
                        // niby najlepsze 193/4/5 ale czasem jak jest duzo niebieskich to moze byc nawet 170/80
   -  contour size Low and High też zależy od zdjęcia tak na prawde ale 15 i 500 to raczej wystarczające granice
'''


# STANDARD
from resources import *
from Klasyfikatory import *



if __name__ == '__main__':
    print(f"Start , using {version} python version")
    start_time = time()

## IMAGES
    # lista adresów do wycików
    dir = "zdjecia_testowe/"
    # lista zdjec
    list_of_images = [dir + img for img in listdir('./{}'.format(dir)) if 'DS' not in img]
    # lista zdjec wycinkow
    # list_of_images = [dir + img for img in os.listdir('./{}'.format(dir)) if img.__contains__('res')]
    print(f'\nImages in {dir} directory : ', *list_of_images, sep='\n'), print('\n')

    # img = imread('Zdjecia/wycinek_5.jpg')
    img_path = list_of_images[1]
    print(f"Chosen image {img_path}")
    img = imread(img_path)


## ALGORITHM
    # creating set of parameters which will be given to segmentation main for finding cells and
    parameters = Parameters(img_path=img_path, thresholdRange=36, thresholdMaskValue=20, CannyGaussSize=3, CannyGaussSigma=0.6,
                            CannyLowBoundry=0.1, CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=False,
                            CannySharpen=False, contourSizeLow=7, contourSizeHigh=500, whiteCellBoundry=186,
                            returnOriginalContours=False)
    segmentation_results = main(parameters)
    print("--- Segmentation completed ---")

## CLASSIFICATION
    ## Supervised methods
    black_path = "./Reference/black/"
    blue_path = "./Reference/blue/"
    blackKNN, blueKNN = KNN(segmentation_results.cells, black_path, blue_path,
                            load_reference_coordinates_path_black='./KNN_black_reference_coordicates.json',
                            load_reference_coordinates_path_blue='./KNN_blue_reference_coordicates.json',
                            working_state='load data')
    blackSVC, blueSVC = classification_using_svc(segmentation_results.cells, black_path, blue_path,
                                                 model_path='./image_classification.model', imageResize=15)
    blackCNN, blueCNN = cnn_classifier(segmentation_results.cells, black_path, blue_path, working_state='load model')

    # Unsupervised methods
    # We use Kmeans two times it gives best results
    blackKmeans, blueKmeans, centroids = kMeans(num_of_clusters=2, cells=segmentation_results.cells)
    kblack, kblue, cent = kMeans(num_of_clusters=2, cells=blueKmeans)
    blueKmeans = kblue
    blackKmeans = blackKmeans + kblack
    # my simple method based on red part of mean RGB values
    black, blue = simple_color_classyfication(segmentation_results.cells)

    # print(f" KNN :: Black {len(blackKNN)} and blue {len(blueKNN)}  /n Finale result of algorithm is  ::  "
    #       f"{len(blackKNN)/(len(blueKNN) + len(blackKNN))*100} % \n")
    print(f" SVC :: Black {len(blackSVC)} and blue {len(blueSVC)}  /n Finale result of algorithm is  ::  "
          f"{len(blackSVC)/(len(blueSVC) + len(blackSVC))*100} % \n")
    print(f" CNN :: Black {len(blackCNN)} and blue {len(blueCNN)}  /n Finale result of algorithm is"
          f"  ::  {len(blackCNN)/(len(blueCNN) + len(blackCNN))*100} % \n")
    print(f" Kmeans :: Black {len(blackKmeans)} and blue {len(blueKmeans)}  /n Finale result of algorithm is  ::  "
          f"{len(blackKmeans)/(len(blueKmeans) + len(blackKmeans))*100} % \n")
    print(f" simple color classification :: Black {len(black)} and blue {len(blue)}  /n Finale result of algorithm is"
          f"  ::  {len(black)/(len(blue) + len(black))*100} % \n")

    print("--- %s seconds ---" % (time() - start_time), ' time after algorithm ')


## SAVE CELLS after SEGMENTATION
    # SAVE Cells in ./Cells  or   ../Reference
    # save_dir = "./Cells"
    # save_dir = "../Reference/"
    # save_cells(segmentation_results.cells, segmentation_results.coordinates, name_addition=f'#{img_path.split("/")[-1]}', dir=save_dir)


## SAVE CELLS after CLASSYFICATION
    # save_cells(blackKNN, coordinates=None, name_addition=f'#', dir='Cells/black')
    # save_cells(blueKNN, coordinates=None, name_addition=f'#', dir='Cells/blue')


# DISPLAY IMGAE WITH CONTOURS
#     # Draw Contours
    drawContours(segmentation_results.image, segmentation_results.contours, -1, (0, 255, 0), 3)
#     # śDisplay
    plot_photo('image', segmentation_results.image)

    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))
    # exit()
