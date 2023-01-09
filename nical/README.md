
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

To count events with a frame containing a sub-string in their call stack

```
cargo run ./tmp/vs-dump.txt count SUBSTRINGTOMATCH
```


# Parameters

- To ignore frames, add new lines to `skipped_frames.txt`.
- To rename a frame into something else add lines to `renamed_frames.txt`
- To skip entire sub-trees, add a the root of the skipped subtree to `skipped_subtrees.txt`
- Annotate events by adding lines in a file (for example `event_annotations.txt`) and passing `-a event_annotations.txt`.

You can add `# comments` to these files, they will be ignored, leading and trailoing whitespaces are also trimmed.

# To generate the dump from visual studio

Used [Erich's steps](https://hackmd.io/HYBz0_5pTAeP2QFG46R_YA#Reproduction-steps) with the print action:

```
#### Ref count for device was modified ({*(unsigned __int64*)(*(unsigned __int64*)(0x000001e212a38ef0 + 0x130) + 8)})\n$CALLSTACK
```

(Replace 0x000001e212a38ef0 with the address you got from mDevice in Erich's steps)

