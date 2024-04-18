python ./compile.py -R 128 test-conversion 1
python ./compile.py -R 128 test-conversion 10
python ./compile.py -R 128 test-conversion 100
python ./compile.py -R 128 test-conversion 1000
python ./compile.py -R 128 test-conversion 10000

python ./compile.py -R 128 test-conversion-with-dabits 1
python ./compile.py -R 128 test-conversion-with-dabits 10
python ./compile.py -R 128 test-conversion-with-dabits 100
python ./compile.py -R 128 test-conversion-with-dabits 1000
python ./compile.py -R 128 test-conversion-with-dabits 10000

python ./compile.py -R 128 test-conversion-with-a2b 1
python ./compile.py -R 128 test-conversion-with-a2b 10
python ./compile.py -R 128 test-conversion-with-a2b 100
python ./compile.py -R 128 test-conversion-with-a2b 1000
python ./compile.py -R 128 test-conversion-with-a2b 10000

python ./compile.py -l -R 128 -C -K LTZ test-conversion-with-fss 1
python ./compile.py -l -R 128 -C -K LTZ test-conversion-with-fss 10
python ./compile.py -l -R 128 -C -K LTZ test-conversion-with-fss 100
python ./compile.py -l -R 128 -C -K LTZ test-conversion-with-fss 1000
python ./compile.py -l -R 128 -C -K LTZ test-conversion-with-fss 10000