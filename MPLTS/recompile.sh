current_dir=$(pwd)

if [ -d "$current_dir/taso" ]; then
  cd taso
fi

rm -rf build
mkdir build
cd build
cmake ..
sudo make install -j 8

cd ../python
python setup.py install

cp /usr/local/lib/libtaso_runtime.so /root/anaconda3/lib/