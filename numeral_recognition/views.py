from django.views.generic import FormView
from .forms import ImageFileForm
from django.http import JsonResponse
from numeral_recognition.numeral_recognition import recognize


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
