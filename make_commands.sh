#!/bin/bash

rm commands

bash permutation_combination.sh 5 4 c | while read line
do
  line=${line/1/contrast}
  line=${line/2/dissimilarity}
  line=${line/3/homogeneity}
  line=${line/4/asm}
  line=${line/5/correlation}

  echo "study anfis --epochs=20 --video=1 --feature=${line}" >> commands
  echo "study anfis --epochs=20 --video=2 --feature=${line}" >> commands
  echo "study anfis --epochs=20 --video=3 --feature=${line}" >> commands
  echo "study anfis --epochs=20 --video=4 --feature=${line}" >> commands
done

