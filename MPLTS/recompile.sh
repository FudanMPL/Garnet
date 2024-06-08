current_dir=$(pwd)

if [ -d "$current_dir/MPLTS" ]; then
  cd MPLTS
fi

sudo rm -rf build
mkdir build
cd build
cmake ..
sudo make install -j 8

cd ../python
python setup.py install

sudo cp /usr/local/lib/libtaso_runtime.so /home/lx/anaconda3/envs/Garnet/lib/