from glob import glob
from setuptools import setup

package_name = 'green_nav_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    description='Green line following node and launches',
    entry_points={
        'console_scripts': [
            'green_nav = green_nav:main'
        ],
    },
)
