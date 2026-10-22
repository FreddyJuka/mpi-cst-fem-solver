# FindMETIS.cmake
# Finds METIS (Ubuntu's libmetis-dev ships no CMake config).
# Hint: METIS_ROOT (CMake variable or environment variable).
# Result: imported target METIS::METIS; variables METIS_FOUND, METIS_INCLUDE_DIR,
# METIS_LIBRARY, METIS_VERSION.

find_path(METIS_INCLUDE_DIR metis.h
  HINTS "${METIS_ROOT}" "$ENV{METIS_ROOT}"
  PATH_SUFFIXES include)
find_library(METIS_LIBRARY metis
  HINTS "${METIS_ROOT}" "$ENV{METIS_ROOT}"
  PATH_SUFFIXES lib lib64)

if(METIS_INCLUDE_DIR AND EXISTS "${METIS_INCLUDE_DIR}/metis.h")
  file(STRINGS "${METIS_INCLUDE_DIR}/metis.h" _metis_ver REGEX "#define METIS_VER_(MAJOR|MINOR|SUBMINOR)")
  string(REGEX REPLACE ".*METIS_VER_MAJOR[ \t]+([0-9]+).*" "\\1" _maj "${_metis_ver}")
  string(REGEX REPLACE ".*METIS_VER_MINOR[ \t]+([0-9]+).*" "\\1" _min "${_metis_ver}")
  string(REGEX REPLACE ".*METIS_VER_SUBMINOR[ \t]+([0-9]+).*" "\\1" _sub "${_metis_ver}")
  set(METIS_VERSION "${_maj}.${_min}.${_sub}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS
  REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR
  VERSION_VAR METIS_VERSION)

if(METIS_FOUND AND NOT TARGET METIS::METIS)
  add_library(METIS::METIS UNKNOWN IMPORTED)
  set_target_properties(METIS::METIS PROPERTIES
    IMPORTED_LOCATION "${METIS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}")
endif()
mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)
