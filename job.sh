#!/bin/bash
i=2
while [ $i -lt 10 ]; do
  echo $i
  python -m nn.softmax $i
  python -m nn.validate $i
  let i+=1
done
