from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import TemplateView


class PortfolioView(TemplateView):
    template_name = 'portfolio.html'

    def get(self, request, *args, **kwargs):
        context = super(PortfolioView, self).get_context_data(**kwargs)
        return render(self.request, self.template_name, context)
