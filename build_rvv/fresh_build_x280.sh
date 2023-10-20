#!/bin/bash

# This file is part of Eigen, a lightweight C++ template library
# for linear algebra.
#
# Copyright (C) 2023, Microchip Technology Inc
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


export EIGEN_MAKE_ARGS='-j16'
./clean.sh

cmake -DCMAKE_TOOLCHAIN_FILE=./cmake.riscv -DCMAKE_INSTALL_PREFIX=./install ..
make install
#make basicstuff
./buildtests.sh packetmath_*
./buildtests.sh special_*
./buildtests.sh fastmath

