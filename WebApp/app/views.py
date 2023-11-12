from django.shortcuts import render, HttpResponse
from django.conf import settings
from .forms import ImageForm
from .models import Image
import os

acceptable_extensions = ['jpg', 'jpeg', 'pdf', 'png']
context = {
    'image_name': '',
    'form': '',
    'image': ''
}


# Create your views here.
def home(request):
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
        #------------------------------------------------------------------------------------------------------------

    return render(request, 'homepage.html', {'context': context})


def doc(request):
    return render(request, 'documentation.html')
