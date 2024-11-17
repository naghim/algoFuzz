import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('requirements-dev.txt') as f:
    extra_required = f.read().splitlines()

setuptools.setup(
    name='algofuzz',
    version='0.1.1',
    author='naghim',
    author_email='naghi.mirtill@gmail.com',
    description='Framework for popular fuzzy c-means clustering algorithms from literature',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/naghim/algofuzz',
    install_requires=required,
    extras_require={
        'dev': extra_required,
        'docs': [],  # please insert here the necessary packages for docs
        'gpu': ["jax[cuda12]"]
    },
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: System :: Clustering',
    ],
    python_requires='>=3.8',
)
