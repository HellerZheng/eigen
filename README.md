# Eigen for Riscv

## Cross Compiling to run on the X280 target

### Building some basic tests close to X280 implementation
### The following script will generate packetmath_1 ~ packetmath_15, fastmath, special_numbers_1 binary files.

```bash
$ cd build_rvv
$ ./fresh_build_x280.sh
```

### For more tests, please check content of buildtests.sh
** Please DONT try to build all tests once, there are hundreds of build items, will generate more than one thousand tests, the building will take long long time to finish. **

### Executing tests on QEMU

```bash
$ cd test/
$ ./packetmath_7
```

## Clean the builds

```bash
$ cd build_rvv/
$ ./clean.sh
```

## Eigen Provided Documentation

**Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.**

For more information go to http://eigen.tuxfamily.org/.

For ***pull request***, ***bug reports***, and ***feature requests***, go to https://gitlab.com/libeigen/eigen.
