#!/bin/sh
echo looking for events with $1
cargo run -- ./tmp/vs-dump.txt count $1 -a ./event_annotations.txt
