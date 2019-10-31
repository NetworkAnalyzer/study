#!/bin/bash

rm result_integer
rm result1_f1

i=1
IFS=','

cat result1 | while read line
do
  if [[ $(( ${i} % 4 )) -eq 0 ]]
  then
    set -- $line
    echo $2 >> result_integer
  fi

  i=$(( ${i} + 1 ))
done

cat result_integer | while true
do
  read car
  read truck

  if [[ -z "$car" ]] ; then break ; fi

  f1=`echo "scale=5; (${car} + ${truck}) / 2" | bc`
  echo $f1 >> result1_f1
done
