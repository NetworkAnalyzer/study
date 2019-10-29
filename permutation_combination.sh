#!/bin/bash
# permutation_combination.sh

# Version: 2.0
# Author : YongYe <complex.invoke@gmail.com>

arg0=-1
argv=${3}
number=${2}
eval ary=({1..${1}})
length=${#ary[@]}
percom(){ nsloop i ${1} number${2} ${3} ${4} ${5}; }
invoke(){ echo $(percom ${argu} nsloop -1) prtcom $(percom ${argu}); }
permut(){ echo -n "${1} arg${i} ${2} "; (( ${#} != 0 )) && echo -n " length "; }
combin(){ (( ${#} != 0 )) && echo -n "${1} arg$((i+1)) arg${i} length " || echo -n "arg$((i+1)) "; }
prtcom(){ num=0; for i in ${@}; do echo -n ${ary[${!i}]}; (( ++num != number )) && echo -n ,; done; echo; }

nsloop()
{ 
    local arc arg
    arg=${1//arg}
    for((${1}=${2}+1; ${1}<${3}; ++${1})); do 
         if [[ ${1//[0-9]} == arg ]]; then
               for((arc=1; arc!=arg; ++arc)); do
                    (( ${1} == arg${arc} )) && continue 2
               done
         fi
         eval eval \\\$\{{4..${#}}\}
    done
}

case ${3} in
     p) argu="-0 +1 permut" ;;
     c) argu="-1 +0 combin" ;; 
esac

$(invoke)

