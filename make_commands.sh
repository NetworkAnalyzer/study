#!/bin/bash

bash permutation_combination.sh 5 3 c | while read line
do
  line=${line/1/contrast}
  line=${line/2/dissimilarity}
  line=${line/3/homogeneity}
  line=${line/4/asm}
  line=${line/5/correlation}

  echo "study anfis --epochs=20 --video=4 --feature=${line}" >> commands
done

