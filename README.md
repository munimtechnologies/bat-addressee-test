# bat-addressee-test

An open, pre-registered re-test by MV2 of Munim, Inc. of a widely cited claim: that everyday
calls of Egyptian fruit bats carry information about **whom they are addressed to** (Prat, Taub &
Yovel 2016, *Scientific Reports* 6:39419). It is the third test in a series on animal "names"
and addressee labels, with [elephant-names-test](https://github.com/munimtechnologies/elephant-names-test)
(Pardo et al. 2024, *Nature Ecology & Evolution*) and
[marmoset-labels-test](https://github.com/munimtechnologies/marmoset-labels-test) (Oren et al.
2024, *Science*).

**Question.** The 2016 study classified the addressee of each call from its acoustics, within
each emitter, and found accuracy above chance (for example bat F6: balanced accuracy 50% against
33% chance, p < 0.01). Its cross-validation left out one vocalization at a time, so calls from
the same file, interaction and recording day could sit in both the training and the test set.
Calls recorded close together share recording conditions, the bat's state and, often, the
addressee. Does addressee information survive when the test is blocked by file and by day, and
is a blocked test powerful enough to see a real addressee effect of the reported size?

**Status (2026-11-20).** Step 1 only: `describe_bats.py` (job `bat-addressee-describe-v1`)
downloads the two public label files of the dataset (Prat, Taub, Pratt & Yovel 2017,
*Scientific Data* 4:170143; figshare collection 3666502, CC0), describes every column, and
counts how emitters, addressees, contexts, files, days and recording channels are arranged. It
also reads only the central directory of each audio archive (no audio is downloaded). It
computes no acoustic statistic and nothing about addressee information. The test itself will be
pre-registered (design, null, power check, decision rules and grader) after this description
and before any acoustic feature is computed.

**Why this is open.** We searched for a re-analysis of the addressee result with blocked
cross-validation and found none (2026-11-20). If we missed one, please open an issue.

## Code

* `describe_bats.py`: the label-only description (stdlib Python).
* `test_describe_bats.py`: its tests (`python -m unittest -v test_describe_bats`).

Data are not redistributed here; the job fetches them from figshare at run time.

Made with AI agents working in MV2's name; no human expert has checked this yet. Code
Apache-2.0; text CC BY 4.0. MV2, Munim, Inc., mv2@munimtech.com.
