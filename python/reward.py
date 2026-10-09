#!/usr/bin/env python3
"""Default GRPO reward for wasmtune.

Fixes two things at once.

1. The reward used to be

        reward_fn = lambda prompts, completions, **kw: [min(len(c) / 500, 1.0)
                                                       for c in completions]

   which rewards *long* answers. That is the exact opposite of the brevity
   doctrine every other part of this package is built around (short answers,
   extraction not regurgitation), so GRPO would cheerfully teach a small model
   to pad.

2. grpo.rewardFile was documented as "./rewards.mjs" and was never actually
   loaded: train_unsloth.py reached for importlib, which cannot import a .mjs,
   and nothing ever passed the path through. GRPO therefore always ran on the
   length reward above. The path is now wired and must point at a *python*
   module.

The replacement scores the two things the rest of the pipeline already scores:

  recall    how many of the reference answer's distinctive terms the completion
            contains. Same signal as src/eval/score.mjs, so a run that improves
            under this reward improves the eval number too.
  brevity   a completion far longer than its reference is rambling, which is
            the failure mode the dataset stage is designed to prevent.

Why reward factual recall rather than something cleverer: the failure mode this
package actually hits is not "the model answers badly", it is "the model never
saw the fact enough times to store it". A recall reward moves every generation
step toward the reference wording, which is the only lever that operates on
facts the corpus contains. It cannot recover facts that are not in the corpus;
nothing reward-shaped can.

This mirrors src/eval/score.mjs rather than importing it, because it runs inside
the python training venv where node is not a dependency.

Entry point is `reward(prompts, completions, **kw)`, the signature TRL's
GRPOTrainer calls. `score()` is the per-sample core if you want it in a script.

Substitute your own with grpo.rewardFile -> a .py file exporting reward().
"""
import re

_STOP = set("""the a an and or for with from that this these those into using used when
which what does have has are was were will would should could there their they
them then than such also only just like over under between through during each
other more most some onto upon within without about after before site website
page pages explain key points answer""".split())

# numbers (incl. 1,234 / 3.10 / v1.2), `code spans`, and Capitalised terms
_RE = re.compile(r"(`[^`]{1,40}`|\b\d[\d,._-]*\b|\b[A-Z][A-Za-z0-9_.!/+:-]{2,})")


def extract_keywords(reference, max_terms=8):
    """Distinctive terms in a reference answer: numbers, code spans and
    Capitalised terms first (as in extractKeywords), then distinctive long
    words."""
    text = str(reference or "")
    found, seen = [], set()
    for m in _RE.finditer(text):
        k = m.group(0).replace("`", "").lower()
        if k and k not in seen and 2 <= len(k) and len(found) < max_terms:
            seen.add(k)
            found.append(k)
    for w in re.split(r"[^a-z0-9_]+", text.lower()):
        if len(found) >= max_terms:
            break
        if len(w) > 5 and w not in _STOP and w not in seen:
            seen.add(w)
            found.append(w)
    return found


def score(prompt, completion, reference=None, max_terms=8):
    """Reward for one sample. Higher is better; 0.0 for an empty completion."""
    completion = str(completion or "")
    if not completion.strip():
        return 0.0
    lower = completion.lower()

    if reference and str(reference).strip():
        kws = extract_keywords(str(reference), max_terms=max_terms)
        if kws:
            recall = sum(1 for k in kws if k.lower() in lower) / len(kws)
        else:
            # A reference with nothing distinctive in it: judge brevity alone.
            recall = 1.0
        # Ramble penalty: past 3x the reference length a completion is a dump.
        ref_len = max(1, len(str(reference)))
        length_factor = 0.5 if len(completion) > ref_len * 3 else 1.0
        return round(recall * length_factor, 4)

    # No reference to recall against (a host calling score() directly): brevity
    # only, which still beats rewarding length.
    return round(min(1.0, 200.0 / max(1, len(completion))), 4)


def reward(prompts, completions, reference=None, references=None, **kw):
    """TRL entry point: batched over prompts/completions.

    The `reference` column comes from the GRPO seed rows (see sftToGrpoSeed),
    so a fact the corpus contains is rewarded when it survives into the
    completion. Accepts either name and either a list or a scalar.
    """
    refs = references if references is not None else reference
    if refs is not None and not isinstance(refs, (list, tuple)):
        refs = [refs]
    refs = refs or []
    out = []
    for i, (p, c) in enumerate(zip(prompts, completions)):
        out.append(score(p, c, refs[i] if i < len(refs) else None))
    return out
