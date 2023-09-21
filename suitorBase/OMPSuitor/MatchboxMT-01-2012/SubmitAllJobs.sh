#!/bin/ksh

for d in 16 8 4 2 1; do

  OMP_NUM_THREADS=$d
  echo "**************************"
  echo $OMP_NUM_THREADS
  echo "**************************" 
  export OMP_NUM_THREADS
  
  ./driverForColoring -infile Rmat_24_Color.rmat | tee ColorResultsRmat.$d
  ./driverForColoring -infile Er_24_Color.rmat | tee ColorResultsEr.$d
  ./driverForColoring -infile Hamrle3.gr -graph dimacs | tee ColorResultsHamrle.$d
  ./driverForColoring -infile cage14.gr -graph dimacs | tee ColorResultsCage.$d
  ./driverForColoring -infile audikw_1.gr -graph dimacs | tee ColorResultsAudi.$d

done

