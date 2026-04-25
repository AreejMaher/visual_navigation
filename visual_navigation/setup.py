from setuptools import find_packages, setup

package_name = 'visual_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rejo',
    maintainer_email='areejmaher57@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'depth_estimator = visual_navigation.depth_estimator:main',
            'motion_tracking = visual_navigation.motion_tracking:main',
            'camera_stream = visual_navigation.camera_stream:main',
            'visual_odometry = visual_navigation.visual_odometry:main',
            'object_detector = visual_navigation.object_detector:main',
            'roi_feature_extraction = visual_navigation.roi_feature_extraction:main',
            'navigation_decision = visual_navigation.navigation_decision:main',
            'action_execution = visual_navigation.action_execution:main',
        ],
    },
)
