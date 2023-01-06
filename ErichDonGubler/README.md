# `antileak`: Erich's attempt to make a tool that assists with refcount issues

This tool assumes you've already set up everything to reproduce the issues it's intended to
investigate according to [this document][hackmd].

[hackmd]: https://hackmd.io/HYBz0_5pTAeP2QFG46R_YA

Two things you'll need:

1. Logs generated from with specific messages attached to breakpoints. The format of messages you'll
	need is:

	* On the refcount tracking start, something like:
	```
	--!! Starting to track refs for `ID3DDevice` at {(unsigned __int64*)(*(unsigned __int64*)((char*)mDevice + 0x130) + 8)}:$CALLSTACK
	```

	* On refcount data breakpoint hit, something like:
	```
	--!! Ref count was changed for {(unsigned __int64*)0x000001f36fcdfae8}:$CALLSTACK
	```
2. Save the logs you generate into a folder with the following structure:

	```
	-- some-folder/
	 |
	 +-- vs-output-window.txt
	 +-- config.toml
	```

	Erich's current contents of `config.toml` are:

	```toml
	count_at_start = 2

	[[stack_matcher]]
	classification = "increment"
	[[stack_matcher.top]]
	type = "symbolicated"
	module = "d3d11.dll"
	symbol_name = "TComObject<NOutermost::CDevice>::AddRef"

	[[stack_matcher]]
	classification = "decrement"
	[[stack_matcher.top]]
	type = "symbolicated"
	module = "d3d11.dll"
	symbol_name = "TComObject<NOutermost::CDevice>::Release"

	[[stack_matcher]]
	classification = "destructor"
	[[stack_matcher.top]]
	type = "symbolicated"
	module = "d3d11.dll"
	symbol_name = "TComObject<NOutermost::CDevice>::~TComObject<NOutermost::CDevice>"
	```

Use `cargo run`. It should answer your questions, and if it's not...then that's
a bug!
