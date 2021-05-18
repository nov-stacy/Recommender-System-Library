from setuptools import setup


packages = {
    'recommender_systems': 'recommender_systems/'
}

setup(
    name='recommender_systems',
    version='0',
    author='Novichkova Anastasia',
    license='MIT',
    description='This library implements algorithms for recommender systems',
    packages=packages,
    package_dir=packages,
    include_package_data=False,
    install_requires=[
        'numpy==1.20.3',
        'pandas==1.2.4',
        'scikit-learn==0.24.2',
        'scipy==1.6.3',
        'threadpoolctl==2.1.0',
        'tqdm==4.60.0',
        'utilspie==0.1.0'
    ]
)
