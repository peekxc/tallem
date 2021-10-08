
## Why is tallem structured the way it is? 

I'm following the package structure layed out: 

- https://www.benjack.io/2018/02/02/python-cpp-revisited.html

Loosely, the benefits of this layout are as follows: 
- Packages used in testing scripts are de-coupled from the main package. This is desirable for dependency minimization; not all
	packages imported in test scripts necessarily should be required to import the package itself.


## Why is meson used to build the extension modules instead of cmake?

I've used CMake for years, and though apparently modern cmake syntax is ok, the DSL is just too clunky to me. 
It's simply not very transparent. The documentation is also quite poor.

Meson is admittedly arguably at a higher-level of abstraction than CMake since it actually can (unofficially)
build CMakeLists based subprojects. That being said, the syntax is just so much more readable and concise. Additionally, 
the documentation of meson is quite good, and the default configurations for many typical target types 'just work.' 
Indeed, in configuring meson.build, I've often find that the final solution ends up being the simplest one--this has
never been the case for me with CMake. 



