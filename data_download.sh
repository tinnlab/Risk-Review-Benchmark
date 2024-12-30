#!/bin/bash

sudo apt-get install unzip
sudo apt-get install wget

# Download the processed data to run the benchmark
mkdir AllData & cd ./AllData
wget "https://seafile.tinnguyen-lab.com/f/47e91f3f051c4abb94a9/?dl=1" -O Data_csv.zip
wget "https://seafile.tinnguyen-lab.com/f/e632d345233a440d9e69/?dl=1" -O Data_rds.zip
wget "https://seafile.tinnguyen-lab.com/f/10be4999104d484bb643/?dl=1" -O Others.zip
wget "https://seafile.tinnguyen-lab.com/f/33543635347e48e6b84f/?dl=1" -O TF-ProcessData.zip

# Unzip the data and remove the zip files
mkdir ReviewPaper_Data 
unzip Data_csv.zip -d ./ReviewPaper_Data
rm Data_csv.zip

mkdir RP_Data_rds_meth25k
unzip Data_rds.zip -d ./RP_Data_rds_meth25k
rm Data_rds.zip

unzip Others.zip
rm Others.zip

mkdir TF-ProcessData
unzip TF-ProcessData.zip -d ./TF-ProcessData
rm TF-ProcessData.zip

cd ..
