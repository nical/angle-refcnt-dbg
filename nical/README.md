
# Running

To list the 10 frames of each stack where the ref count was modified:

```
cargo run ./tmp/vs-dump.txt list 10
```

To print as a tree:

```
cargo run ./tmp/vs-dump.txt tree
```

There is an example of the generated tree in `tmp/tree.txt`

# Parameters

To ignore frames, add new lines to `skipped_frames.txt`.
To rename a frame into something else add lines to `renamed_frames.txt`

# To generate the dump from visual studio

Used [Erich's steps](https://hackmd.io/HYBz0_5pTAeP2QFG46R_YA#Reproduction-steps) with the print action:

```
#### Ref count for device was modified ({*(unsigned __int64*)(*(unsigned __int64*)(0x000001e212a38ef0 + 0x130) + 8)})\n$CALLSTACK
```

(Replace 0x000001e212a38ef0 with the address you got from mDevice in Erich's steps)

