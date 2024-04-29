from setuptools import setup, find_packages

setup(
    name='yt_tools',
    version='1.1.1',
    packages=find_packages(),
    include_package_data=True,  # 这确保包数据被包括在安装中
    package_data={
        # 包含AIA_response目录下的所有.npz文件和根目录下的config.json文件
        'yt_tools': ['AIA_response/*.npz', 'config.json'],
    },
    description='analysis the AMRVAC spherical data',
    long_description='Analysis of mhd simulation data in three-dimensional spherical coordinates of amrvac',
    author='Fu Wentai, Chen Guoyin, Ni Yiwei',
    author_email='gychen@smail.nju.edu.cn',
    install_requires=[
        'numpy>=1.18.5', 
        'yt>=3.6.0',      
        'matplotlib>=3.2.2',  
        'scipy>=1.5.0',   
        'aglio>=0.2.0',
        'vtk>=9.3.0',
        'plotly>=5.20.0'
    ],
)
