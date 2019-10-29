#!/bin/bash

cat commands | while read line
do
  $line >> accuracy
done

