from django.shortcuts import render

# Create your views here.


def post_index(request):
    return render(request, 'velo/index.html', {})
