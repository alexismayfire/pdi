from setuptools import setup

setup(
    name='tattoo_ar',
    version='0.1.0',
    packages=['tattoo_ar'],
    entry_points={
        'console_scripts': [
            'tattoo_ar = tattoo_ar.__main__:main'
        ]
    },
)