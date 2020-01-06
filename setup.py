from setuptools import setup, find_packages

setup(
    name='Spin Cavity Dyanmics',
    version='1.0.0',
    url='https://github.com/mypackage.git',
    author='Author Name',
    author_email='jszabo94@gmail.com',
    description='''Create simple spin system coupled to an external environment.
                    Used to observe how nonequilibrium dynamics of fundamental
                    models such as Ising, Heisenber, Kitaev, etc are impacted or
                    lead to novel signatures in the coupled environment. Uses QuSpin
                    package for setting up ED calculations and Krylov time evolution,
                    and cavity QED analysis techniques in QuTip package.''',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1'],
)
