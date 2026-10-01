"""
Setup script for DTCWT Popularity Assessment
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "DTCWT-based Data Popularity Assessment for Distributed Systems"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'PyWavelets>=1.1.1',
        'dtcwt>=0.12.0',
        'scikit-learn>=1.0.0',
        'statsmodels>=0.13.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
    ]

setup(
    name='dtcwt-popularity',
    version='4.0.2',
    author='Sajjad Pirahesh',
    author_email='pirahesh@phd.tabrizu.ac.ir',
    description='DTCWT-based Data Popularity Assessment for Distributed Systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/s-pirahesh/dtcwt_popularity',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.12',
    install_requires=requirements,
    extras_require={
        'deep-learning': ['tensorflow>=2.8.0'],
        'tracking': ['mlflow>=2.0.0'],
        'dev': [
            'jupyter>=1.0.0',
            'pytest>=6.2.0',
            'ipython>=7.30.0',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt'],
    },
    keywords=[
        'popularity prediction',
        'caching',
        'distributed systems',
        'wavelet transform',
        'DTCWT',
        'time series analysis',
    ],
)
