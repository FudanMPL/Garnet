#!/bin/sh

echo '# Getting started' > readme.md
sed -e '1 d' ../README.md >> readme.md
