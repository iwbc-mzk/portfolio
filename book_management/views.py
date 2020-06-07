from django.shortcuts import render
from django.views.generic import TemplateView


# Create your views here.
class HomeView(TemplateView):
    template_name = 'book_management/home.html'

    def get(self, request, *args, **kwargs):
        context = super(HomeView, self).get_context_data(**kwargs)
        return render(self.request, self.template_name, context)
