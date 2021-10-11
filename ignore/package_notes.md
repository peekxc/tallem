
## Why is tallem structured the way it is? 

I'm following the package structure layed out: 

- https://www.benjack.io/2018/02/02/python-cpp-revisited.html

Loosely, the benefits of this layout are as follows: 
- Packages used in testing scripts are de-coupled from the main package. This is desirable for dependency minimization; not all packages imported in test scripts necessarily should be required to import the package itself.


## Why is meson used to build the extension modules instead of cmake?

After using CMake for years, I've finally decided that it is just not a build system I want to continue using. The official documentation is awful, what's considered "best practice" is effectively folklore, and the DSL is ugly. It is simply not a very transparent build system to me. I would place CMakes DSL quite high on the list of offenders of the [principle of least astonishment](https://en.wikipedia.org/wiki/Principle_of_least_astonishment).  

That being said, CMake is popular and nowadays [almost] the de-facto standard for cross-platform C++ build systems. Supposedly modern cmake syntax is actually quite o.k. if you take the time to learn it, and there are mature (unofficial) documentation sites that have accumulated some notion how to write "good CMake." This is a heavily disputed area though. 

In contrast, [Mesons]() DSL is just beautiful. Admittedly, Meson places more complexity on the build pipeline _if_ your project includes subprojects that are themselves CMake projects (see [their philosophy on mixing build systems](https://mesonbuild.com/Mixing-build-systems.html)). That being said, the syntax is very easy to follow, the documentation is quite good, and the default configurations for many typical target types *just work*. Indeed, in configuring a meson.build, I've often find that the final solution that works across the CI platforms ends up being the simplest one; . So learning how to write a "good meson.build configuration" seems to me to be a learning process that converges on simplicity. This has *never* been the case for me with CMake. 

For these reasons and more, `tallem`'s extension modules are built outside of setuptools with Meson. 

Is the use of Meson--a very clearly C++ oriented build system---for Python package management an unwise choice? With the dawning of PEP 514/515 and the move towards a more diversified build system for Python packaging, I would argue not so. And I am not alone. Indeed, SciPy appears to be [moving towards a Meson build configuration](https://labs.quansight.org/blog/2021/07/moving-scipy-to-meson/). 

### Is `tallem` a python application? 

No, `tallem` is a python _library_. The distinction is important, see: https://caremad.io/posts/2013/07/setup-vs-requirement/. Thus, `tallem` does not have a `requirements.txt` for use with, e.g. `Pipenv`.





