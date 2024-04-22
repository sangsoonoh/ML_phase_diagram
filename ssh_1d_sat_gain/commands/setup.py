from setuptools import setup
setup(
    name='ssh_1d_commands',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'c=commands:cli',
            'configure-cpp-functions=commands:configure_cpp',
            'build-cpp-functions=commands:build_cpp',
        ]
    }
)