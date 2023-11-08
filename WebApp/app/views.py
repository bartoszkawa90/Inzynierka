from django.shortcuts import render, HttpResponse
from django.conf import settings
import os

acceptable_extensions = ['jpg', 'jpeg', 'pdf', 'png']

# Create your views here.
def home(request):
    if request.method == 'POST':

        # print(f'Received files \n {request.FILES}')

        # if request.FILES["file_upload"].name.split('.')[1] not in acceptable_extensions:
        #     print('[ERROR] Wrong type of file were selected')
        # else:
        ## SAVE IMAGE WITH ITS NAME
        # save_path = os.path.join(settings.MEDIA_ROOT,
        #                          request.FILES["file_upload"].name)

        ## ZAPISUJEMY Z TA SAMĄ NAZWĄ ZEBY NADPISAĆ STARE ZDJECIE
        save_path = 'image.jpg'

        with open(save_path, "wb") as output_file:
            for chunk in request.FILES["file_upload"].chunks():
                output_file.write(chunk)

    return render(request, 'homepage.html')

def doc(request):
    return render(request, 'documentation.html')
