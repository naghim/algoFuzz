import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='algofuzz',
    version='0.0.1',
    author='naghim',
    author_email='naghi.mirtill@gmail.com',
    description='Framework for popular fuzzy c-means clustering algorithms from literature',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/naghim/algofuzz',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: System :: Clustering',
    ],
    python_requires='>=3.6',
)