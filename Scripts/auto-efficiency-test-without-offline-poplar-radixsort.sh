python3 ./Scripts/data_prepare_for_efficiency.py IRIS
./Scripts/ring-without-offline.sh -pn 10000 poplar-radixsort-efficiency-IRIS

python3 ./Scripts/data_prepare_for_efficiency.py wine
./Scripts/ring-without-offline.sh -pn 10000 poplar-radixsort-efficiency-wine

python3 ./Scripts/data_prepare_for_efficiency.py cancer
./Scripts/ring-without-offline.sh -pn 10000 poplar-radixsort-efficiency-cancer

python3 ./Scripts/data_prepare_for_efficiency.py tic-tac-toe
./Scripts/ring-without-offline.sh -pn 10000 poplar-radixsort-efficiency-tic-tac-toe

python3 ./Scripts/data_prepare_for_efficiency.py kohkiloyeh
./Scripts/ring-without-offline.sh -pn 10000 poplar-radixsort-efficiency-kohkiloyeh

python3 ./Scripts/data_prepare_for_efficiency.py diagnosis
./Scripts/ring-without-offline.sh -pn 10000 poplar-radixsort-efficiency-diagnosis

python3 ./Scripts/data_prepare_for_efficiency.py digits
./Scripts/ring-without-offline.sh -pn 10000 poplar-radixsort-efficiency-digits

python3 ./Scripts/data_prepare_for_efficiency.py adult
./Scripts/ring-without-offline.sh -pn 10000 poplar-radixsort-efficiency-adult

