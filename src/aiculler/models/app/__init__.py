"""Code ported from the image-triage app (tylermcm/ImageTriage @ e1eeaf842b0147644cbb4800566a1ac8ea8635c0).

* ``maniqa_net.py``: ``image_triage/cascade/rankers/_maniqa_net.py``, verbatim (itself adapted from IIGROUP/MANIQA,
  Apache-2.0).
* ``blink.py``: the pure functions of ``image_triage/cascade/blink.py``, verbatim.
* ``faces.py``: the pure geometry of ``image_triage/cascade/faces.py``, verbatim.

The engines that use them are in ``aiculler.models.engines`` (from ``cascade/rankers/worker.py``). Keep these
files in step with the app: a change there is a change in what the culler measures (bump the scorer revision).
"""
