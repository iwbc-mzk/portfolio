import os
from django import forms


class ImageFileForm(forms.Form):
    ext_list = ['.jpg', '.png']
    img = forms.FileField(label='')

    def clean_img(self):
        img = self.cleaned_data.get('img')
        _, ext = os.path.splitext(str(img))

        if ext not in self.ext_list:
            raise forms.ValidationError('使用できる拡張子は[jpg, png]のみです')
        elif img.size > 1048576:
            raise forms.ValidationError('使用できる画像は1Mbまでです')
        else:
            return img
