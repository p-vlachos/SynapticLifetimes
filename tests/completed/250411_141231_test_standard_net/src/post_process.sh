#!/bin/bash

NPARSIM=1
NCORES=1
MEMGB=2
LOCAL_COMPUTE=false
DEBUG=false
CLUSTER='x-men'

while getopts "hln:c:m:ds" opt; do
    case $opt in
    h) echo "usage: $0 [-h] [-a] [-l] ..."; exit ;;
    l) LOCAL_COMPUTE=true ;;
    n) NPARSIM=$OPTARG ;;
    c) NCORES=$OPTARG ;;
    m) MEMGB=$OPTARG ;;
    d) DEBUG=true ;;
    s) CLUSTER='sleuths' ;;
    \?) echo "error: option -$OPTARG is not implemented"; exit ;;
    esac
done


cd ../

if $LOCAL_COMPUTE
then
   echo "doing local"
   python -m src.net.xstrct_run -x -c $NPARSIM -t
else
   echo "doing nonlocal"
   srun -p $CLUSTER -c $NCORES --mem $MEMGB --time 29-00 python \
	-m src.net.xstrct_run -x -c $NPARSIM -t
fi
