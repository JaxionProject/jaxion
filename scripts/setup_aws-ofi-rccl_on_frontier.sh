#!/bin/bash

# Set up AWS-OFI-RCCL plugin for better performance.
# See https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html

rocm_version=6.2.4

# Load modules
module load PrgEnv-gnu/8.6.0
module load rocm/$rocm_version
module load craype-accel-amd-gfx90a
module load gcc-native/13.2
module load cray-mpich/8.1.31
libfabric_path=/opt/cray/libfabric/1.22.0

# change to outside the jaxion directory
cd ../
rm -rf aws-ofi-rccl

# Download the plugin repo
git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
cd aws-ofi-rccl

# Build the plugin
./autogen.sh
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/hip/lib:$LD_LIBRARY_PATH
PLUG_PREFIX=$PWD

CC=hipcc CFLAGS=-I/opt/rocm-$rocm_version/include ./configure \
--with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
--prefix=$PLUG_PREFIX --with-hip=/opt/rocm-$rocm_version/hip --with-mpi=$MPICH_DIR

make
make install

# Reminder to export the plugin to your path
echo $PLUG_PREFIX
echo "Add the following line in the environment to use the AWS OFI RCCL plugin"
echo "export LD_LIBRARY_PATH="$PLUG_PREFIX"/lib:$""LD_LIBRARY_PATH"
