# DAHRS

# FCFA Framework  
**Function-word Cross-lingual Framework for Alignment**

A Python framework for analyzing cross-lingual word alignments with a focus on **function words**, **semantic role labeling (SRL)**, and **alignment divergence patterns**.

# licenses?

---

## Overview

The **FCFA Framework** provides tools for analyzing and improving cross-lingual word alignments. It emphasizes linguistic structure by incorporating:

- Function word analysis  
- Semantic Role Labeling (SRL)  
- Divergence detection (one-to-many, many-to-one)  

This framework is designed for **NLP research**, **machine translation analysis**, and **alignment debugging**.

---

## Why FCFA?

Traditional alignment methods often struggle with:

- Function word mismatches across languages  
- One-to-many and many-to-one alignments  
- Loss of semantic structure  

**FCFA addresses these challenges by:**

- Explicitly modeling function words  
- Integrating SRL tags into alignment analysis  
- Providing interpretable visualization tools  

---

## Quick Start

```python
from fcfa_framework import FCFAFramework

framework = FCFAFramework()

src_tokens = {0: "Hello", 1: "world", 2: "!"}
tgt_tokens = {0: "Bonjour", 1: "le", 2: "monde", 3: "!"}

framework.load_alignment_output("0-0 1-1 1-2 2-3", (src_tokens, tgt_tokens))

print(framework.get_alignment_statistics())
print("One-to-many:", framework.find_one_to_many())
print("Many-to-one:", framework.find_many_to_one())
