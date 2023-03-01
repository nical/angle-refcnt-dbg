* I've got a lot of fixing of the configuration language to do. I changed `antileak.kdl` to be what
	I want it to be. Now, I just need `config` parsing to adhere to it.
* The model for compressing tails and matching operations is now a set of matchers that run at every
	point on cloned event iterators. This is powerful, and shouldn't be too much of an adjustment
	from current code. Currently thinking of applying the following groups in order:

	* matchers
		* manual
		* locally balanced ops
	* compressors

	I think I'm going to need to adjust the current event model to include activations and
	deactivations of stack frames.

	* New events:
		* Activation (new)
		* Refcount event
			* Start event
			* ModifyEvent + classification (incl. unknown)
		* Deactivation

	Problem: I need classifiers to be a separate layer of producing events for this. Otherwise, the
	code would get weirdly complex.

	* Might want to start interning stack frames for speed?
	* What happens when multiple matchers match? Feels like we should report and error and not make
		a flamegraph yet.
