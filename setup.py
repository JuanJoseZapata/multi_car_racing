from setuptools import setup

setup(
    name='gym_multi_car_racing',
    version='0.0.1',
    url='https://github.com/igilitschenski/multi_car_racing',
    description='Multi Car Racing Environment',
    packages=['gym_multi_car_racing'],
    install_requires=[
        'shapely~=1.7.0',
        'numpy>=1.10.4',
        'gym~=0.17.2',
        'pettingzoo>=1.23.1'
    ]
)