from setuptools import setup

setup(
   name='face-recognition',
   version='0.0.1',
   description='Face recognition',
   author='TuanNN',
   author_email='tuannn0107@gmail.src',
   packages=['src'],  #same as name
   install_requires=['tensorflow',
                     'opencv-python',
                     'dlib',
                     'scikit-learn',
                     'scikit-image',
                     'facenet'], #external packages as dependencies
)