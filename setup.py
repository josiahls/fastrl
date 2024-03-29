from pkg_resources import parse_version
from configparser import ConfigParser
import subprocess
from setuptools.command.develop import develop
from setuptools.command.install import install
import setuptools
import os
assert parse_version(setuptools.__version__)>=parse_version('36.2')

# note: all settings are in settings.ini; edit there, not here
config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']

cfg_keys = 'version description keywords author author_email'.split()
expected = cfg_keys + "lib_name user branch license status min_python audience language".split()
for o in expected: assert o in cfg, "missing expected setting: {}".format(o)
setup_cfg = {o:cfg[o] for o in cfg_keys}

licenses = {
    'apache2': ('Apache Software License 2.0','OSI Approved :: Apache Software License'),
}
statuses = [ '1 - Planning', '2 - Pre-Alpha', '3 - Alpha',
    '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive' ]
py_versions = '3.7 3.8 3.9 3.10'.split()

requirements = ['pip', 'packaging']
if cfg.get('requirements'): requirements += cfg.get('requirements','').split()
if cfg.get('pip_requirements'): requirements += cfg.get('pip_requirements','').split()
dev_requirements = (cfg.get('dev_requirements') or '').split()

lic = licenses[cfg['license']]
min_python = cfg['min_python']

# Define the repository and branch or commit you want to install from
TORCHDATA_GIT_REPO = "https://github.com/josiahls/data.git"
TORCHDATA_COMMIT = "main"  # or replace with a specific commit hash

class CustomInstall(install):
    def run(self):
        # Ensure that torchdata is cloned and installed before proceeding
        print('Cloning torchdata')
        subprocess.check_call(["git", "clone", TORCHDATA_GIT_REPO])
        print('Installing torchdata')
        subprocess.check_call(["pip", "install","-vvv", "./data"])
        # Call the standard install.
        install.run(self)

class CustomDevelop(develop):
    def run(self):
        # Ensure that torchdata is cloned but not installed
        if not os.path.exists('data'):
            print('Cloning torchdata')
            subprocess.check_call(["git", "clone", TORCHDATA_GIT_REPO])
        try:
            import torchdata
        except ImportError:
            print('Installing torchdata')
            subprocess.check_call(["pip", "install","-vvv", "-e", "./data"])
        # Call the standard develop.
        develop.run(self)

setuptools.setup(
    name = cfg['lib_name'],
    license = lic[0],
    classifiers = [
        'Development Status :: ' + statuses[int(cfg['status'])],
        'Intended Audience :: ' + cfg['audience'].title(),
        'License :: ' + lic[1],
        'Natural Language :: ' + cfg['language'].title(),
    ] + ['Programming Language :: Python :: '+o for o in py_versions[py_versions.index(min_python):]],
    url = cfg['git_url'],
    packages = setuptools.find_packages(),
    include_package_data = True,
    install_requires = requirements,
    extras_require={'dev': dev_requirements },
    dependency_links = cfg.get('dep_links','').split(),
    python_requires  = '>=' + cfg['min_python'],
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    entry_points = { 'console_scripts': cfg.get('console_scripts','').split() },
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    },
    **setup_cfg)

