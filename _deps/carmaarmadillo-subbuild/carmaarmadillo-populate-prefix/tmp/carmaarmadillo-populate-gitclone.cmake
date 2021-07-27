
if(NOT "/Users/mpiekenbrock/tallem/_deps/carmaarmadillo-subbuild/carmaarmadillo-populate-prefix/src/carmaarmadillo-populate-stamp/carmaarmadillo-populate-gitinfo.txt" IS_NEWER_THAN "/Users/mpiekenbrock/tallem/_deps/carmaarmadillo-subbuild/carmaarmadillo-populate-prefix/src/carmaarmadillo-populate-stamp/carmaarmadillo-populate-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/Users/mpiekenbrock/tallem/_deps/carmaarmadillo-subbuild/carmaarmadillo-populate-prefix/src/carmaarmadillo-populate-stamp/carmaarmadillo-populate-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/Users/mpiekenbrock/tallem/extern/carma/extern/armadillo-code"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/Users/mpiekenbrock/tallem/extern/carma/extern/armadillo-code'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/local/bin/git"  clone --no-checkout --config "advice.detachedHead=false" "https://gitlab.com/conradsnicta/armadillo-code.git" "armadillo-code"
    WORKING_DIRECTORY "/Users/mpiekenbrock/tallem/extern/carma/extern"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://gitlab.com/conradsnicta/armadillo-code.git'")
endif()

execute_process(
  COMMAND "/usr/local/bin/git"  checkout 10.5.x --
  WORKING_DIRECTORY "/Users/mpiekenbrock/tallem/extern/carma/extern/armadillo-code"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '10.5.x'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/local/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/Users/mpiekenbrock/tallem/extern/carma/extern/armadillo-code"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/Users/mpiekenbrock/tallem/extern/carma/extern/armadillo-code'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/Users/mpiekenbrock/tallem/_deps/carmaarmadillo-subbuild/carmaarmadillo-populate-prefix/src/carmaarmadillo-populate-stamp/carmaarmadillo-populate-gitinfo.txt"
    "/Users/mpiekenbrock/tallem/_deps/carmaarmadillo-subbuild/carmaarmadillo-populate-prefix/src/carmaarmadillo-populate-stamp/carmaarmadillo-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/Users/mpiekenbrock/tallem/_deps/carmaarmadillo-subbuild/carmaarmadillo-populate-prefix/src/carmaarmadillo-populate-stamp/carmaarmadillo-populate-gitclone-lastrun.txt'")
endif()

