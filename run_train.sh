#!/bin/bash

python main.py \
    --task atis \
    --model_type bert \
    --model_dir atis_model \
    --do_train --do_eval
