mkdir -p build
cd build
cmake ..
sleep 0.1
echo "0.1s waited"
make
# cmake --build .
python reference/generate_arccos_data.py