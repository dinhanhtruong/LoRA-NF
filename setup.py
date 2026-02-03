from setuptools import setup

setup(
    name='lora-nf',
    version='0.1.0',    
    description='Low-Rank Adaptation of Neural Fields',
    url='https://github.com/dinhanhtruong/LoRA-NF',
    author='Anh Truong',
    license='MIT License',
    packages=['lora_nf'],
    install_requires=['pytorch',
                      'numpy',                     
                      'imageio',                     
                      'trimesh',                     
                      'pysdf',                     
                      'PyMCubes',                     
                      'packaging'],
    classifiers=[
        'Programming Language :: Python :: 3.12',
    ],
)
