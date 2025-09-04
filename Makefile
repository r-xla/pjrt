.PHONY: format format-cpp format-r

# C++ files: only top-level files in src/, exclude RcppExports.cpp
CXX_SOURCES := $(filter-out src/RcppExports.cpp,$(wildcard src/*.cpp))
CXX_HEADERS := $(wildcard src/*.h)
CXX_FILES   := $(CXX_SOURCES) $(CXX_HEADERS)

# R files: all top-level files in R/ (exclude RcppExports.R) and tests/testthat/*.R
R_FILES       := $(filter-out R/RcppExports.R,$(wildcard R/*.R))
TEST_R_FILES  := $(wildcard tests/testthat/*.R)
ALL_R_FILES   := $(R_FILES) $(TEST_R_FILES)

format: format-cpp format-r

format-cpp:
	@echo "Formatting C/C++ sources with clang-format"
	@if [ -n "$(strip $(CXX_FILES))" ]; then clang-format -i $(CXX_FILES); fi

format-r:
	@echo "Formatting R sources with air"
	@if [ -n "$(strip $(ALL_R_FILES))" ]; then air format $(ALL_R_FILES); fi
