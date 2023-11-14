from django.shortcuts import render, HttpResponse
from django.conf import settings
from .forms import ImageForm
from .models import Image
import os
import cv2

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
def home(request):
    print(f' \n\n\n Home request')
    if request.method == 'POST':
        # print data from POST and FILES
        print(f'POST {request.POST}')
        print(f'FILES {request.FILES}')

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

    # GET PARAMETERS FROM TEXTAREAS ------------------------------------------------------------------------------
    if request.method == 'GET' and 'threshold_range' in request.GET:
        # print data from POST and FILES
        print(f'POST {request.POST}')
        print(f'FILES {request.FILES}')
        print(f'GET {request.GET}')

        # current_params = []
        cur_params = {}
        for idx in range(len(parameters)):
            if request.GET.get(parameters[idx]) == '' or request.GET.get(parameters[idx]) == None:
                cur_params[parameters[idx]] = request.GET.get(default_params[idx])
            else:
                cur_params[parameters[idx]] = int(request.GET.get(parameters[idx]))
        context['cur_params'] = cur_params

        print(f'Current parameters  {cur_params}')
        # saved_image_name = os.listdir("images")[0]
        # saved_image = cv2.imread(os.path.join('images/', saved_image_name))
        # print(f' Image name {saved_image_name}, \n image : {saved_image.shape}')

    return render(request, 'homepage.html', {'context': context})


def doc(request):
    return render(request, 'documentation.html')
