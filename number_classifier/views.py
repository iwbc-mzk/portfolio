from django.views.generic import FormView
from .forms import ImageFileForm
from django.http import JsonResponse
from numeral_recognition.numeral_recognition import recognize
from io import BytesIO
import base64


class NumeralRecognitionForm(FormView):
    template_name = 'numeral_recognition/numeral_recognition.html'
    form_class = ImageFileForm

    def form_valid(self, form):
        response = {}
        if self.request.method == 'POST':
            result = recognize(self.request.FILES['img'])
            response['result'] = str(result)
            return JsonResponse(response)
        else:
            response['error_msg'] = '不正なリクエストです'

    def form_invalid(self, form):
        response = {'error_msg': form.errors['img']}
        return JsonResponse(response)


# 参考: https://qiita.com/conta_/items/047c2ad89f43ade5dab3
def canvas(request):
    response = {}
    base64_img = request.POST['img']
    code = base64.b64decode(base64_img.split(',')[1])
    img = BytesIO(code)
    result = recognize(img)
    response['result'] = str(result)
    return JsonResponse(response)
