# DAHRS  
**Divergence-Aware Hallucination-Remediated SRL Projection**

A Python framework for improving cross-lingual **Semantic Role Labeling (SRL) projection** using divergence-aware alignment and hallucination mitigation techniques.

# license?

---

## Overview

**DAHRS** is a framework for projecting semantic role labels (SRL) across languages while addressing two major challenges:

- **Cross-lingual divergence** (structural and lexical differences)
- **Hallucinated alignments** (incorrect or spurious mappings)

The system builds on alignment analysis techniques and introduces strategies to improve projection quality through:

- Divergence-aware alignment handling  
- Function word filtering  
- Structured alignment resolution (FCFA: First-Come-First-Assign)  

---

## Paper
This repo goes along with the paper:

"DAHRS: Divergence-Aware Hallucination-Remediated SRL Projection"
It covers the ideas, methods, and experiments behind what's behind implemented here.

https://arxiv.org/abs/2407.09283

---

## Key Ideas

### Divergence-Aware Alignment
Captures and analyzes:
- One-to-many alignments  
- Many-to-one alignments  
- Structural mismatches between languages  

---

### Hallucination Remediation
Reduces incorrect alignments by:
- Filtering function words  
- Removing inconsistent mappings  
- Prioritizing stable alignment structures  

---

### FCFA Alignment Strategy
Implements a **First-Come-First-Assign (FCFA)** approach:

- Sequential alignment resolution  
- Conflict reduction in many-to-one mappings  
- Improved consistency in projected labels  

---

## Quick Start

```python
from fcfa_framework import FCFAFramework

framework = FCFAFramework()

src_tokens = {0: "I", 1: "do", 2: "n't", 3: "know"}
tgt_tokens = {0: "Je", 1: "ne", 2: "sais", 3: "pas"}

framework.load_alignment_output("0-0 1-1 2-3 3-2", (src_tokens, tgt_tokens))

print(framework.get_alignment_statistics())
