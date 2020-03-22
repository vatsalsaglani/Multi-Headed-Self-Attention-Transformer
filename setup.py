from setuptools import setup


setup(
	name='MultiHeadedTransformer', 
    version = '0.1', 
    url = 'https://github.com/vatsalsaglani/Multi-Headed-Self-Attention-Transformer',
    description='A packaged code to train a corpus of text with a Multi Headed Self-Attention transformer to generate text.',
	author = 'Vatsal Saglani', 
    license = 'MIT', 
    install_requires = ['torch', 'torchvision', 'tqdm'],
    packages = ['Transformer'],
	zip_safe = False

)
