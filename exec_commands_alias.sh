#!/bin/bash

cat commands_alias | while read line
do
  $line >> accuracy_alias
  echo "${line} is done"
done

