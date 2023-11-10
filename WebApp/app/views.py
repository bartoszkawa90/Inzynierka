from django.shortcuts import render, HttpResponse
from django.conf import settings
import os

acceptable_extensions = ['jpg', 'jpeg', 'pdf', 'png']
context = {
    'image_name': ''
}

# Create your views here.
def home(request):
    if request.method == 'POST':
        ## ZAPISUJEMY Z TA SAMĄ NAZWĄ ZEBY NADPISAĆ STARE ZDJECIE
        save_path = 'image.jpg'
        image_name = request.FILES["file_upload"].name

        with open(save_path, "wb") as output_file:
            for chunk in request.FILES["file_upload"].chunks():
                output_file.write(chunk)
        ## CREATE CONTEXT
        context['image_name'] = image_name

    return render(request, 'homepage.html', {'context': context})# {'image_name': image_name})

def doc(request):
    return render(request, 'documentation.html')
