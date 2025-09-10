from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='ngeodesic',
      version='0.0.1',
      description='ngeodesic',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/ngeodesic-ai/ngeodesic',
      author = "icmoore",
      author_email = "ngeodesic@gmail.com",
      license='MIT',
      package_dir = {"ngeodesic": "python/prod"},
      packages=[
          'ngeodesic',
          'ngeodesic.erc'
      ],   
      zip_safe=False)
