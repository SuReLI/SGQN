cd src/env/dm_control
pip install -e .

cd ../dmc2gym
pip install -e .

cd ../../..

curl https://codeload.github.com/nicklashansen/dmcontrol-generalization-benchmark/tar.gz/main | tar -xz --strip=3 dmcontrol-generalization-benchmark-main/src/env/data
mv data src/env/