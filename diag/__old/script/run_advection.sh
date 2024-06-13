#!/bin/bash


#OMP_NUM_THREADS=192 OMP_PLACES=cores  OMP_PROC_BIND=spread /mnt/source/parallel-advection/build_dpcpp/src/advection /mnt/source/parallel-advection/build_dpcpp/src/advection.ini 2>&1 >/dev/null
#OMP_NUM_THREADS=192 OMP_PLACES=cores  OMP_PROC_BIND=spread /mnt/source/parallel-advection/build_dpcpp/src/advection /mnt/source/parallel-advection/build_dpcpp/src/advection.ini

/mnt/source/parallel-advection/build_dpcpp/src/advection /mnt/source/parallel-advection/build_dpcpp/src/advection.ini 2>&1 >/dev/null
/mnt/source/parallel-advection/build_dpcpp/src/advection /mnt/source/parallel-advection/build_dpcpp/src/advection.ini

for i in {1..2}
do
  OMP_NUM_THREADS=192 OMP_PLACES=cores  OMP_PROC_BIND=spread /mnt/source/parallel-advection/build_dpcpp/src/advection /mnt/source/parallel-advection/build_dpcpp/src/advection.ini
done