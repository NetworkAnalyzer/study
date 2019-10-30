#!/bin/bash

cat commands | while read line
do
  $line >> accuracy
  echo "${line} is done"
done

