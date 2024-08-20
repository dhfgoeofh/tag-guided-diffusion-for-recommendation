from setuptools import setup, find_packages

exec(open('denoising_diffusion_pytorch/version.py').read())

setup(
  name = 'Diffusion Recommender model',
  packages = find_packages(),
  version = __version__,
  license='Dong-A University',
  description = 'Conditional Diffusion model for Recommendation',
  author = 'Daero Kim',
  author_email = 'eofh4817@gmail.com',
  url = '',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'generative models'
  ],
  install_requires=[
    'accelerate',
    'einops',
    'ema-pytorch>=0.4.2',
    'numpy',
    'pillow',
    'pytorch-fid',
    'scipy',
    'torch',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
