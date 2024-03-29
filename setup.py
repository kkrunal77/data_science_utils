from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='data_science_utils',
    url='https://github.com/kkrunal77/data_science_utils',
    author='kkrunal77',
    author_email='k.krunal77@gmail.com',
    # Needed to actually package something
    packages = find_packages(),
    # Needed for dependencies
    install_requires=['numpy','tensorflow','tensorflow_datasets','pandas','more-itertools',
          'dill','seaborn','joblib','opencv-python'],
    # *strongly* suggested for sharing
    # version='0.1',
    # The license can be anything you like
    #license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)