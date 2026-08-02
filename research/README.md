# research/

Reading lists and design notes: the papers behind each mechanism in this
repository, and *why* those mechanisms made the cut.

| File | Contents |
| --- | --- |
| [slm-reading-list.md](slm-reading-list.md) | The small-language-model canon: models, data, scaling |
| [architecture-notes.md](architecture-notes.md) | Paper-by-paper justification for every architectural choice in `src/` |
| [looped-transformers.md](looped-transformers.md) | Design notes on the supra2 looped family |
| [post-training-notes.md](post-training-notes.md) | SFT/CoT/DPO/RLVR/SPIN literature and what transfers to small models |
| [image-model-notes.md](image-model-notes.md) | Diffusion/AR image generation at small scale |

House rule: when a design decision in code cites a paper, the entry here says
what we *took* from it and what we deliberately ignored — a reading list that
doesn't editorialise is just a bibliography.
