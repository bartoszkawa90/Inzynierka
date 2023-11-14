from django.shortcuts import render, HttpResponse
from django.conf import settings
from .forms import ImageForm
from .models import Image
import os
from .resources import *

acceptable_extensions = ['jpg', 'jpeg', 'pdf', 'png']
parameters = ['threshold_range', 'threshold_mask', 'cell_low_size', 'cell_high_size', 'white_cells_boundry']
default_params = [31, 20, 7, 500, 187]
default_params = {'threshold_range': 31, 'threshold_mask': 20, 'cell_low_size': 7, 'cell_high_size': 500, 'white_cells_boundry': 187}
context = {
    'image_name': '',
    'form': '',
    'image': '',
    'cur_params': default_params
}


# Create your views here.
def home(request, parameters=parameters):
    # print post get data
    print(f' \n\n  Home request \n')
    if request.method == 'POST':
        print(f'POST {request.POST}')
        print(f'FILES {request.FILES}')
    if request.method == 'GET':
        print(f'GET {request.GET}')
    print("\n\n")

    if request.method == 'POST' and 'Upload' in request.POST:
        # GET IMAGE AND ITS NAME AND SAVE TO DIR AND DATABASE AFTER CLEARING PREVIOUS IMAGES-------------------------
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            ## CLEAR DATABASE AND IMAGES DIR TO SAVE NEW IMAGE CORRECTLY
            os.system('python manage.py flush --no-input')
            try:
                for file in os.listdir('images'):
                    os.remove(os.path.join('images/', file))
            except:
                print('[ERROR] some error with images in images occured')
            form.save()
        # CREATE CONTEXT AND SAVE IMAGE NAME
        try:
            image_name = request.FILES['image'].name
            images = Image.objects.all()
        except:
            image_name = ''
            images = [None]
        context['image_name'] = image_name
        context['form'] = form
        context['image'] = [images[0]]
        some_images_saved = 1

    elif request.method == 'POST' and 'Find contours' in request.POST:
        print('Find contours')
        # calculate contours
        saved_image_name = os.listdir("images")[0]
        saved_image = imread(os.path.join('images/', saved_image_name))
        new_image = split(saved_image)[2]
        # image = Image.objects.get(id=1)
        # print('Image', image.name)
        # form = ImageForm()
        # form.Meta.model.image = new_image
        # print(f' New form {form.is_valid()}')
        # form.save()

        #
        # parameters = Parameters(img_path=saved_image, thresholdRange=context['cur_params']['threshold_range'],
        #                         thresholdMaskValue=context['cur_params']['threshold_mask'],
        #                         CannyGaussSize=3, CannyGaussSigma=0.6, CannyLowBoundry=0.1, CannyHighBoundry=10.0,
        #                         CannyUseGauss=True, CannyPerformNMS=False,
        #                         contourSizeLow=context['cur_params']['cell_low_size'],
        #                         contourSizeHigh=context['cur_params']['cell_high_size'],
        #                         whiteCellBoundry=context['cur_params']['white_cells_boundry'])
        # segmentation_results = main(parameters)
        # print("--- Segmentation completed ---")


    # elif request.method == 'POST' and 'Calculate' in request.POST:
    #     pass


    # GET PARAMETERS FROM TEXTAREAS ------------------------------------------------------------------------------
    if request.method == 'GET' and 'threshold_range' in request.GET:
        cur_params = {}
        for idx in range(len(parameters)):
            if request.GET.get(parameters[idx]) == '' or request.GET.get(parameters[idx]) == None:
                cur_params[parameters[idx]] = request.GET.get(default_params[idx])
            else:
                cur_params[parameters[idx]] = int(request.GET.get(parameters[idx]))
        context['cur_params'] = cur_params
        print(f'Current parameters  {cur_params}')

    return render(request, 'homepage.html', {'context': context})


def doc(request):
    return render(request, 'documentation.html')
