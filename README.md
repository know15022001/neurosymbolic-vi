# neurosymbolic-vi
## NeuroSymbolic AI Without Matrix Multiplication
*A Graph-Based QA System for Vietnamese — Pure Python, CPU Only, Zero Hallucination*

  `Python 3.10+`       `NumPy only`       `Hallucination: 0%`       `License: CC BY 4.0`  
``  79% accuracy  |  0% hallucination  |  0.057ms inference  |  1.1MB RAM  |  trains in 13ms ``

## What This Is
A question-answering AI system that uses no matrix multiplication, no neural network, and no GPU. It learns from raw Vietnamese text, builds a knowledge graph automatically, and answers questions through ontological reasoning.
If it does not know the answer, it says so. It never fabricates. Every answer comes with a full reasoning trace showing exactly how the conclusion was reached.

Demo: https://colab.research.google.com/drive/1tZvIsRVQqBL-sW_xD6Wm86hu8E8M9jOe?usp=sharing

## Key Results
| Metric |	This System |	N-Gram Baseline |	Tiny LSTM Baseline |
| :----- | :----------- | :---------------| :------------------|
| Keyword Accuracy |	79% |	38% |	0% |
| Hallucination Rate |	0% |	Yes |	Yes |
| Unknown Detection |	100% |	0% |	0% |
| Avg Inference Time |	0.057 ms |	0.102 ms |	5.5 ms |
| RAM After Training |	1.1 MB |	1.2 MB |	15.5 MB |
| Train Time (523 sentences) |	0.013 s |	0.008 s |	9.5 s |
| Explainability |	100% |	0% |	0% |
| Update Without Retrain |	Yes |	No |	No |

## Quick Start

>git clone https://github.com/know15022001/neurosymbolic-vi  
>cd neurosymbolic-vi  
>pip install numpy psutil
>
>python benchmark_v3.py

That is all. No GPU. No CUDA. No PyTorch. No transformers. Runs in under 15 seconds on any modern CPU.

## Try It Yourself  
>from core import VocabGraph, KnowledgeDB, LearningEngine, InferenceEngine, CompoundTokenizer``
>
>tokenizer = CompoundTokenizer()  
>vocab = VocabGraph()  
>kb = KnowledgeDB()  
>le = LearningEngine(vocab, kb, tokenizer=tokenizer)  
>  
>le.learn_sentence('cats eat fish.')  
>le.learn_sentence('a cat is a pet.')  
>le.learn_sentence('pets are cute.')  
>  
>engine = InferenceEngine(vocab, kb, tokenizer=tokenizer)  
>result = engine.process('what does a cat eat?')  
>print(result['answer'])         # Cats eat fish.  
>print(result['reasoning_trace']) # Full step-by-step trace

## Repository Structure
| File |	Description |
| :--- | :--------- |
| core.py |	Main engine: VocabGraph, KnowledgeDB, LearningEngine, InferenceEngine, CompoundTokenizer |
| baselines.py |	N-Gram (trigram+backoff) and Tiny LSTM baselines for comparison |
| benchmark_v3.py |	Full benchmark: all three models, 18 test cases, side-by-side output |
| data/corpus_vi.txt |	523-sentence Vietnamese training corpus (animals, geography, science) |
| requirements.txt |	numpy, psutil — nothing else |

## How It Works
### Five Components, No Matrices
* Vocab Graph — stores words as nodes, relations as typed weighted edges (is_a, has, can, lives_in)
* Knowledge DB — stores facts as {subject, predicate, object, confidence} tuples with source counting
* CompoundTokenizer — discovers compound words (e.g. 'thong minh' = intelligent) using PMI statistics, no labels needed
* Grammar Rules — detects query intent (ask_object, ask_definition, ask_property, ask_location) via ordered regex patterns
* Reasoning KB — controls ontology climbing with confidence decay (0.88 per level) and UNKNOWN threshold (0.20)

### Ontology Climbing
When a direct fact is not found, the system climbs the is_a hierarchy:
Query: 'what does a calico cat eat?'
Step 1: lookup(calico cat, eat) -> NOT FOUND
Step 2: calico cat is_a cat -> lookup(cat, eat) -> FOUND: fish, meat (conf=0.79)
Answer: A calico cat eats fish, meat (inherited from 'cat').

## Authorship
Conceptual design, system architecture, and all research direction: Anonymous User.
Implementation, debugging, benchmarking, and analysis: Claude (Anthropic).
See the paper for a full discussion of the human-AI collaborative research process.

## Citation
>@article{neurosymbolic_vi_2026,  
>  title   = {NeuroSymbolic AI Without Matrix Multiplication:  
>             A Graph-Based Approach to Knowledge Representation},  
>  author  = {Anonymous User and Claude (Anthropic)},  
>  year    = {2026},  
>  url     = {https://github.com/know15022001/neurosymbolic-vi}  
>}

## License
CC BY 4.0 — free to use, share, and adapt with attribution.
