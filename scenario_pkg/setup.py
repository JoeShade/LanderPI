from glob import glob
from setuptools import setup

package_name = 'scenario_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # Install the original scripts so the runner can invoke them from the installed share.
        ('share/' + package_name, ['line_following.py', 'green_nav.py', 'HRI.py']),
        # Shared audio directory next to scenario_runner.py for all nodes.
        ('share/' + package_name + '/scenario_pkg/feedback_voice', glob('scenario_pkg/feedback_voice/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Scenario Maintainer',
    maintainer_email='maintainer@example.com',
    description='Mission scenario orchestrator combining line following, green beacon pursuit, and HRI gestures.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scenario_runner = scenario_pkg.scenario_runner:main',
        ],
    },
)
