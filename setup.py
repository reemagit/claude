from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='pyclaude',
    url='https://github.com/reemagit/claude',
    author='Enrico Maiorino',
    author_email='enrico.maiorino@gmail.com',
    # Needed to actually package something
    packages=['pyclaude'],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'tqdm'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Exponential Random Graph models as null models for network analysis',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)