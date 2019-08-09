#!/bin/zsh
kill -9  $(ps aux | grep train | grep hancock | grep -v grep | awk {'print$2'})
