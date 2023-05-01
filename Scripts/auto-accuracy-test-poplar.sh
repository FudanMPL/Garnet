python3 ./Scripts/data_prepare_for_accuracy.py IRIS
./Scripts/ring-without-offline.sh poplar-accuracy-IRIS

python3 ./Scripts/data_prepare_for_accuracy.py wine
./Scripts/ring-without-offline.sh poplar-accuracy-wine

python3 ./Scripts/data_prepare_for_accuracy.py cancer
./Scripts/ring-without-offline.sh poplar-accuracy-cancer

python3 ./Scripts/data_prepare_for_accuracy.py tic-tac-toe
./Scripts/ring-without-offline.sh poplar-accuracy-tic-tac-toe

python3 ./Scripts/data_prepare_for_accuracy.py kohkiloyeh
./Scripts/ring-without-offline.sh poplar-accuracy-kohkiloyeh

python3 ./Scripts/data_prepare_for_accuracy.py diagnosis
./Scripts/ring-without-offline.sh poplar-accuracy-diagnosis

python3 ./Scripts/data_prepare_for_accuracy.py digits
./Scripts/ring-without-offline.sh poplar-accuracy-digits

python3 ./Scripts/data_prepare_for_accuracy.py adult
./Scripts/ring-without-offline.sh poplar-accuracy-adult

