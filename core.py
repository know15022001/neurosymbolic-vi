"""
NeuroSymbolic AI - Core Implementation
Kiến trúc: Graph + Rules + Ontological Inference
Không dùng ma trận hay neural network
"""

import re
import time
import json
import math
from collections import defaultdict, Counter
from typing import Optional


# ─────────────────────────────────────────────
#  COMPOUND TOKENIZER (PMI-based)
# ─────────────────────────────────────────────

class CompoundTokenizer:
    """
    Tự khám phá từ ghép tiếng Việt bằng PMI.
    Học từ corpus → nhận ra 'thông minh', 'nguy hiểm',
    'dễ thương', 'thú cưng', 'động vật'... là 1 đơn vị.
    Giải quyết vấn đề compound word bị tách thành token lẻ.
    """
    STOP = {
        'là','có','và','của','với','trong','ngoài','trên','dưới',
        'được','để','cho','bằng','hay','hoặc','mà','thì','rất',
        'đều','cũng','vì','do','từ','tới','đến','khi','nếu',
        'thường','luôn','khá','hơi','đã','đang','sẽ','một','các',
        'những','này','đó','kia','ấy','gì','ai','đâu','sao','không',
        'theo','về','qua','sau','trước','giữa','nên','cần','phải',
    }
    PMI_BI  = 4.0
    PMI_TRI = 5.0
    MIN_COUNT = 1  # corpus nhỏ → 1 lần đủ nếu PMI cao

    def __init__(self):
        self.compound_set = set()   # set of compound strings
        self._token_count = Counter()
        self._total = 0
        self._fitted = False

    def fit(self, sentences: list):
        """Học compound words từ corpus bằng PMI."""
        ngram_count = Counter()
        self._token_count = Counter()

        for sent in sentences:
            toks = self._raw_tok(sent)
            self._token_count.update(toks)
            # Bigram (bỏ stop ở cả hai vị trí)
            for i in range(len(toks)-1):
                if toks[i] not in self.STOP and toks[i+1] not in self.STOP:
                    ngram_count[(toks[i], toks[i+1])] += 1
            # Trigram (không có stop ở bất kỳ vị trí nào)
            for i in range(len(toks)-2):
                tri = toks[i:i+3]
                if not any(t in self.STOP for t in tri):
                    ngram_count[tuple(tri)] += 1

        self._total = sum(self._token_count.values())
        if self._total == 0:
            return

        self.compound_set = set()
        for ngram, cnt in ngram_count.items():
            if cnt < self.MIN_COUNT:
                continue
            score = self._pmi(ngram, cnt)
            if len(ngram) == 2 and score >= self.PMI_BI:
                self.compound_set.add(' '.join(ngram))
            elif len(ngram) == 3 and score >= self.PMI_TRI:
                self.compound_set.add(' '.join(ngram))

        self._fitted = True

    def tokenize(self, text: str) -> list:
        """
        Tách câu thành tokens, ghép compound words lại.
        Ưu tiên trigram > bigram > unigram (greedy left-to-right).
        """
        toks = self._raw_tok(text)
        if not self._fitted:
            return toks
        result = []; i = 0
        while i < len(toks):
            if i <= len(toks)-3:
                tri = toks[i]+' '+toks[i+1]+' '+toks[i+2]
                if tri in self.compound_set:
                    result.append(tri); i += 3; continue
            if i <= len(toks)-2:
                bi = toks[i]+' '+toks[i+1]
                if bi in self.compound_set:
                    result.append(bi); i += 2; continue
            result.append(toks[i]); i += 1
        return result

    def _pmi(self, ngram: tuple, cnt: int) -> float:
        p = cnt / self._total
        p_indep = 1.0
        for w in ngram:
            p_indep *= self._token_count[w] / self._total
        return math.log(p / p_indep) if p_indep > 0 else 0

    def _raw_tok(self, text: str) -> list:
        return [t for t in re.sub(r'[^\w\s]',' ', text.lower()).split() if t]

    def stats(self):
        return {
            'compounds': len(self.compound_set),
            'bigrams': sum(1 for c in self.compound_set if len(c.split())==2),
            'trigrams': sum(1 for c in self.compound_set if len(c.split())==3),
        }


# ─────────────────────────────────────────────
#  DATA STORES
# ─────────────────────────────────────────────

class VocabGraph:
    """
    Lưu từ vựng và quan hệ giữa các từ
    {id, key, relationships: [{target_id, type, weight}]}
    """
    def __init__(self):
        self.nodes = {}          # id -> {key, pos_hints, count}
        self.edges = defaultdict(list)  # id -> [{target_id, type, weight}]
        self.key_to_id = {}      # key -> id
        self._counter = 0

    def add_or_get(self, key: str) -> str:
        if key in self.key_to_id:
            self.nodes[self.key_to_id[key]]['count'] += 1
            return self.key_to_id[key]
        vid = f"V{self._counter:05d}"
        self._counter += 1
        self.nodes[vid] = {'key': key, 'count': 1, 'pos_hints': []}
        self.key_to_id[key] = vid
        return vid

    def add_edge(self, from_key: str, to_key: str, rel_type: str, weight: float = 1.0):
        fid = self.add_or_get(from_key)
        tid = self.add_or_get(to_key)
        # Tăng weight nếu đã có edge này
        for e in self.edges[fid]:
            if e['target_id'] == tid and e['type'] == rel_type:
                e['weight'] += weight
                return
        self.edges[fid].append({'target_id': tid, 'type': rel_type, 'weight': weight})

    def get_relations(self, key: str, rel_type: str = None):
        vid = self.key_to_id.get(key)
        if not vid:
            return []
        edges = self.edges[vid]
        if rel_type:
            edges = [e for e in edges if e['type'] == rel_type]
        return [(self.nodes[e['target_id']]['key'], e['type'], e['weight']) for e in edges]

    def get_parents(self, key: str):
        """Trả về các node cha (is_a relationship)"""
        return [k for k, t, w in self.get_relations(key, 'is_a')]

    def stats(self):
        return {'nodes': len(self.nodes), 'edges': sum(len(v) for v in self.edges.values())}


class KnowledgeDB:
    """
    Lưu facts với confidence score
    {subject, predicate, object, confidence, source_count}
    """
    def __init__(self):
        self.facts = []
        self._index = defaultdict(list)  # (subject, predicate) -> [fact_idx]
        self.synonyms = defaultdict(set) # word -> {synonyms}

    def add(self, subject: str, predicate: str, obj: str, confidence: float = 1.0):
        key = (subject, predicate)
        # Tăng confidence nếu đã có fact này
        for i in self._index[key]:
            if self.facts[i]['object'] == obj:
                self.facts[i]['source_count'] += 1
                self.facts[i]['confidence'] = min(0.99,
                    self.facts[i]['confidence'] + 0.05)
                return
        idx = len(self.facts)
        self.facts.append({
            'subject': subject, 'predicate': predicate,
            'object': obj, 'confidence': confidence, 'source_count': 1
        })
        self._index[key].append(idx)

    def query(self, subject: str, predicate: str = None) -> list:
        # BUG3 FIX: case-insensitive subject lookup
        subjects_to_try = {subject, subject.lower()}
        if predicate:
            results = []
            for s in subjects_to_try:
                idxs = self._index.get((s, predicate), [])
                results.extend([self.facts[i] for i in idxs])
            return sorted(results, key=lambda x: -x['confidence'])
        results = []
        for (s, p), idxs in self._index.items():
            if s in subjects_to_try:
                results.extend([self.facts[i] for i in idxs])
        return sorted(results, key=lambda x: -x['confidence'])

    def add_synonym(self, w1: str, w2: str):
        """X và Y đều giống nhau → lưu làm synonym 2 chiều."""
        w1, w2 = w1.lower().strip(), w2.lower().strip()
        if w1 and w2 and w1 != w2:
            self.synonyms[w1].add(w2)
            self.synonyms[w2].add(w1)

    def get_synonyms(self, word: str) -> set:
        return self.synonyms.get(word.lower(), set())

    def query_with_synonyms(self, subject: str, predicate: str) -> list:
        """Query thử cả synonyms của predicate value."""
        results = self.query(subject, predicate)
        if results:
            return results
        # Thử synonym của predicate
        syn_results = []
        for (s, p), idxs in self._index.items():
            if s in {subject, subject.lower()} and p == predicate:
                for i in idxs:
                    obj = self.facts[i]['object']
                    syns = self.get_synonyms(obj)
                    # Trả về fact với note là synonym
                    f = dict(self.facts[i])
                    f['via_synonym'] = obj
                    syn_results.append(f)
        return syn_results

    def stats(self):
        return {'facts': len(self.facts),
                'synonym_groups': len(self.synonyms)}


class GrammarRules:
    """
    Patterns ngôn ngữ và intent detection
    """
    # Prefix danh từ tiếng Việt cần normalize
    NOUN_PREFIXES = re.compile(
        r'^(con|cái|cây|con con|chiếc|cái con|loài|loại)\s+', re.IGNORECASE)
    TRAILING_CO = re.compile(r'\s+có$', re.IGNORECASE)

    RULES = [
        # BUG2 FIX: 'có + VERB + không' → ask_object_yn (đặt TRƯỚC ask_property)
        # VD: "chó có ăn không?" → hỏi chó có ăn hay không (khác hỏi tính chất)
        {'pattern': r'(.+)\s+có\s+(ăn|uống|làm|học|chơi|bay|bơi|chạy|sống)\s+không',
         'intent': 'ask_verb_yn', 'subject_group': 1, 'verb_group': 2},
        # Câu hỏi về object của hành động — "X ăn gì?"
        {'pattern': r'(.+)\s+(ăn|uống|làm|học|chơi|dùng|thích)\s+gì',
         'intent': 'ask_object', 'subject_group': 1, 'verb_group': 2},
        # Câu hỏi định nghĩa
        {'pattern': r'(.+)\s+là\s+gì',
         'intent': 'ask_definition', 'subject_group': 1},
        # Câu hỏi tính từ đặt TRƯỚC ask_property để bắt trường hợp cụ thể
        {'pattern': r'(.+)\s+(đáng yêu|dễ thương|nguy hiểm|thông minh|đẹp|ngon|tốt|xấu|khỏe|nhanh|chậm|lớn|nhỏ|trung thành)\s+không',
         'intent': 'ask_adjective', 'subject_group': 1, 'adj_group': 2},
        # Câu hỏi có/không (tính chất chung) — đặt SAU các rule cụ thể hơn
        {'pattern': r'(.+)\s+có\s+(.+)\s+không',
         'intent': 'ask_property', 'subject_group': 1, 'property_group': 2},
        # Câu hỏi ở đâu
        {'pattern': r'(.+)\s+ở\s+đâu',
         'intent': 'ask_location', 'subject_group': 1},
        # Câu hỏi tại sao
        {'pattern': r'tại sao\s+(.+)',
         'intent': 'ask_reason', 'subject_group': 1},
        # Fact pattern: X là Y
        {'pattern': r'^(.+)\s+là\s+(.+)\.',
         'intent': 'fact_is_a', 'subject_group': 1, 'object_group': 2},
        # Fact pattern: X có Y
        {'pattern': r'^(.+)\s+có\s+(.+)\.',
         'intent': 'fact_has', 'subject_group': 1, 'object_group': 2},
        # Fact pattern: X ăn Y
        {'pattern': r'^(.+)\s+(ăn|uống)\s+(.+)\.',
         'intent': 'fact_eats', 'subject_group': 1, 'verb_group': 2, 'object_group': 3},
        # Fact pattern: X sống ở/trong/gần Y
        {'pattern': r'^(.+)\s+sống\s+(ở|trong|gần|dưới|trên)\s+(.+)\.',
         'intent': 'fact_lives_in', 'subject_group': 1, 'object_group': 3},
    ]

    @classmethod
    def normalize_subject(cls, subject: str) -> str:
        """
        BUG1 FIX: Bỏ prefix 'con/cái/cây...' để normalize subject.
        'con chó' → 'chó', 'cây táo' → 'táo', 'con mèo tam thể' → 'mèo tam thể'
        """
        s = cls.NOUN_PREFIXES.sub('', subject).strip()
        s = cls.TRAILING_CO.sub('', s).strip()
        return s

    @classmethod
    def match(cls, text: str,
              tokenizer: 'CompoundTokenizer' = None) -> Optional[dict]:
        # BUG3 FIX: lowercase để match case-insensitive
        text = text.strip()
        text_lower = text[0].lower() + text[1:] if text else text

        for rule in cls.RULES:
            m = re.match(rule['pattern'], text_lower, re.IGNORECASE)
            if not m:
                m = re.match(rule['pattern'], text, re.IGNORECASE)
            if m:
                result = {'intent': rule['intent'], 'raw': text}
                if 'subject_group' in rule:
                    raw_subj = m.group(rule['subject_group']).strip()
                    # Normalize: bỏ prefix con/cái, lowercase
                    result['subject'] = cls.normalize_subject(raw_subj).lower()
                if 'object_group' in rule:
                    result['object'] = m.group(rule['object_group']).strip()
                if 'verb_group' in rule:
                    result['verb'] = m.group(rule['verb_group']).strip()
                if 'property_group' in rule:
                    result['property'] = m.group(rule['property_group']).strip()
                if 'adj_group' in rule:
                    result['adjective'] = m.group(rule['adj_group']).strip()
                return result
        return None


class ReasoningKB:
    """
    Meta-rules về cách suy luận
    """
    CONFIDENCE_DECAY = 0.88      # per ontology level
    CONFIDENCE_THRESHOLD = 0.20  # dưới ngưỡng này → không biết
    MAX_ONTOLOGY_LEVELS = 5
    SUBJECTIVE_ADJECTIVES = {
        'đáng yêu', 'dễ thương', 'đẹp', 'xấu', 'ngon',
        'tốt', 'thú vị', 'hay', 'buồn', 'vui', 'đáng sợ'
    }

    @classmethod
    def is_subjective(cls, term: str) -> bool:
        return any(adj in term.lower() for adj in cls.SUBJECTIVE_ADJECTIVES)


# ─────────────────────────────────────────────
#  LEARNING ENGINE
# ─────────────────────────────────────────────

class LearningEngine:
    """
    Tự học từ data thô — không cần labels
    """
    # Patterns để infer relationship type
    IS_A_PATTERNS = [
        r'(.+?)\s+là\s+(một\s+)?(.+?)[\.\,]',
        r'(.+?)\s+là\s+loài\s+(.+?)[\.\,]',
        r'(.+?)\s+là\s+động vật\s+(.+?)[\.\,]',
    ]
    HAS_PATTERNS = [
        r'(.+?)\s+có\s+(.+?)[\.\,]',
    ]
    CAN_PATTERNS = [
        r'(.+?)\s+có thể\s+(.+?)[\.\,]',
        r'(.+?)\s+(ăn|uống|sống|bay|bơi|chạy|kêu|leo)\s+(.+?)[\.\,]',
    ]
    LIVES_IN_PATTERNS = [
        r'(.+?)\s+sống\s+(ở|trong|gần|dưới|trên)\s+(.+?)[\.\,]',
    ]

    # Prefix cần bỏ khi normalize subject trong corpus
    _SUBJ_NORM = re.compile(
        r'^(con|cái|cây|chiếc|loài|loại|một)\s+', re.IGNORECASE)

    def _norm(self, s: str) -> str:
        """Normalize subject: bỏ prefix, lowercase, strip"""
        s = self._SUBJ_NORM.sub('', s.strip())
        return s.lower().strip()

    def __init__(self, vocab: VocabGraph, knowledge: KnowledgeDB,
                 tokenizer: 'CompoundTokenizer' = None):
        self.vocab = vocab
        self.knowledge = knowledge
        self.tokenizer = tokenizer
        self.sentences_learned = 0

    def learn_sentence(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:
            return

        self.sentences_learned += 1

        # Thêm tất cả tokens vào vocab
        tokens = self._tokenize(sentence)
        for tok in tokens:
            self.vocab.add_or_get(tok)

        # Extract is_a relationships
        for pattern in self.IS_A_PATTERNS:
            m = re.match(pattern, sentence, re.IGNORECASE)
            if m:
                subj = self._norm(m.group(1))
                obj = m.group(m.lastindex).strip()
                if subj and obj and subj != obj:
                    obj = re.sub(r'^(một|loài|loại)\s+', '', obj)
                    # FIX: Truncate object tại verb phụ (được/để/cho/bằng/mà/khi)
                    obj = re.split(r'\s+(được|để|cho|bằng|mà|khi|nhằm|mà)\s+', obj)[0].strip()
                    # Nếu object quá dài (>3 token) → chỉ lấy 2 token đầu
                    obj_toks = obj.split()
                    if len(obj_toks) > 3:
                        obj = ' '.join(obj_toks[:2])
                    if subj and obj:
                        self.vocab.add_edge(subj, obj, 'is_a')
                        self.knowledge.add(subj, 'is_a', obj, 0.9)
                break

        # Extract has relationships
        for pattern in self.HAS_PATTERNS:
            m = re.match(pattern, sentence, re.IGNORECASE)
            if m:
                subj = self._norm(m.group(1))
                obj = m.group(2).strip()
                # Bỏ câu hỏi "có ... không"
                if 'không' not in obj and subj and obj:
                    self.vocab.add_edge(subj, obj, 'has')
                    self.knowledge.add(subj, 'has', obj, 0.85)
                break

        # Extract can/action relationships
        for pattern in self.CAN_PATTERNS:
            m = re.match(pattern, sentence, re.IGNORECASE)
            if m:
                subj = self._norm(m.group(1))
                if m.lastindex >= 3:
                    verb = m.group(2).strip()
                    obj = m.group(3).strip()
                    if subj and obj:
                        self.vocab.add_edge(subj, f"{verb} {obj}".strip(), 'can')
                        self.knowledge.add(subj, 'can', f"{verb} {obj}".strip(), 0.8)
                elif m.lastindex == 2:
                    action = m.group(2).strip()
                    if subj and action:
                        self.vocab.add_edge(subj, action, 'can')
                        self.knowledge.add(subj, 'can', action, 0.8)
                break

        # Extract lives_in
        for pattern in self.LIVES_IN_PATTERNS:
            m = re.match(pattern, sentence, re.IGNORECASE)
            if m:
                subj = self._norm(m.group(1))
                loc = m.group(3).strip() if m.lastindex >= 3 else m.group(2).strip()
                if subj and loc:
                    self.vocab.add_edge(subj, loc, 'lives_in')
                    self.knowledge.add(subj, 'lives_in', loc, 0.85)
                break

        # Extract property/adjective facts — dùng compound tokenizer
        # Pattern: [subject] [adverb?] [adjective] .
        ADVERBS = {'rất','thường','luôn','hay','khá','hơi','vô cùng','cực kỳ'}
        if self.tokenizer and self.tokenizer._fitted:
            ctoks = self.tokenizer.tokenize(sentence.rstrip('.'))
            # Lọc bỏ adverb, lấy subject và adj
            filtered = [t for t in ctoks if t not in ADVERBS]
            if len(filtered) >= 2:
                # Token cuối có thể là adj, token đầu là subject
                last = filtered[-1]
                # Chỉ nhận adj nếu là compound (2+ âm tiết) hoặc trong KNOWN_ADJ
                KNOWN_ADJ = {'đẹp','ngon','tốt','xấu','lớn','nhỏ',
                             'nhanh','chậm','lạnh','nóng','ngọt',
                             'chua','cay','đắng','mặn','độc','dài','rộng'}
                is_compound_adj = ' ' in last   # compound word
                is_known_adj = last in KNOWN_ADJ
                if is_compound_adj or is_known_adj:
                    subj_toks = filtered[:-1]
                    # Bỏ stop words khỏi subject
                    SUBJ_STOP = {'là','có','được','để','và','của'}
                    subj_toks = [t for t in subj_toks if t not in SUBJ_STOP]
                    if subj_toks:
                        subj = self._norm(' '.join(subj_toks[-2:]))
                        if subj and subj != last:
                            self.knowledge.add(subj, 'property', last, 0.8)
        else:
            # Fallback regex nếu không có tokenizer
            prop_match = re.match(
                r'^(.+?)\s+(rất\s+)?(đáng yêu|nguy hiểm|thông minh|đẹp|ngon|tốt|trung thành|nhanh|chậm|lớn|nhỏ|dễ thương|quý giá)\b',
                sentence, re.IGNORECASE)
            if prop_match:
                subj = self._norm(prop_match.group(1))
                adj = prop_match.group(3).strip()
                if subj:
                    self.knowledge.add(subj, 'property', adj, 0.8)

        # Extract synonyms: "X và Y đều giống nhau"
        syn_match = re.search(
            r"(.+?)\s+và\s+(.+?)\s+đều\s+(giống nhau|như nhau|tương tự)",
            sentence, re.IGNORECASE)
        if syn_match:
            w1 = self._norm(syn_match.group(1))
            w2 = self._norm(syn_match.group(2))
            if w1 and w2:
                self.knowledge.add_synonym(w1, w2)

        # Extract membership: "X gồm Y và Z" → Y is_a X, Z is_a X
        mem_match = re.search(
            r"(.+?)\s+(?:gồm|bao gồm|gồm có)\s+(.+)",
            sentence, re.IGNORECASE)
        if mem_match:
            parent = self._norm(mem_match.group(1))
            members_raw = mem_match.group(2).rstrip(".")
            members = re.split(r"\s+và\s+|\s*,\s*", members_raw)
            for mem in members:
                mem_n = self._norm(mem)
                if mem_n and parent and mem_n != parent:
                    self.vocab.add_edge(mem_n, parent, "is_a")
                    self.knowledge.add(mem_n, "is_a", parent, 0.85)

        # Extract eats/drinks
        eat_match = re.match(r'^(.+?)\s+(ăn|uống)\s+(.+?)[\.\,]', sentence, re.IGNORECASE)
        if eat_match:
            subj = self._norm(eat_match.group(1))
            verb = eat_match.group(2).strip()
            obj = eat_match.group(3).strip()
            if subj and obj:
                self.knowledge.add(subj, verb, obj, 0.9)

    def learn_corpus(self, sentences: list[str]):
        # Fit compound tokenizer trên toàn bộ corpus trước
        if self.tokenizer is not None:
            self.tokenizer.fit(sentences)
        for sent in sentences:
            self.learn_sentence(sent)

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer is not None and self.tokenizer._fitted:
            return self.tokenizer.tokenize(text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t.strip() for t in text.split() if t.strip()]


# ─────────────────────────────────────────────
#  INFERENCE ENGINE
# ─────────────────────────────────────────────

class InferenceEngine:
    """
    Pipeline: Parse → Lookup → Ontology Climb → Generate
    """

    def __init__(self, vocab: VocabGraph, knowledge: KnowledgeDB,
                 tokenizer: 'CompoundTokenizer' = None):
        self.vocab = vocab
        self.knowledge = knowledge
        self.tokenizer = tokenizer

    def process(self, query: str) -> dict:
        t0 = time.perf_counter()
        trace = []

        parsed = GrammarRules.match(query, self.tokenizer)
        if not parsed:
            parsed = self._fallback_parse(query)
            trace.append(f"[PARSE] Không match pattern — dùng fallback parse")
        else:
            trace.append(f"[PARSE] Intent: {parsed['intent']}, Subject: {parsed.get('subject','?')}")

        result = self._dispatch(parsed, trace)

        elapsed = (time.perf_counter() - t0) * 1000
        result['inference_time_ms'] = round(elapsed, 3)
        result['reasoning_trace'] = trace
        result['query'] = query
        return result

    def _dispatch(self, parsed: dict, trace: list) -> dict:
        intent = parsed.get('intent', 'unknown')
        subject = parsed.get('subject', '')

        if intent == 'ask_definition':
            return self._answer_definition(subject, trace)
        elif intent == 'ask_object':
            return self._answer_object(subject, parsed.get('verb', 'ăn'), trace)
        elif intent in ('ask_property', 'ask_adjective'):
            prop = parsed.get('property') or parsed.get('adjective', '')
            return self._answer_property(subject, prop, trace)
        elif intent == 'ask_location':
            return self._answer_location(subject, trace)
        else:
            return self._answer_general(subject, trace)

    def _answer_definition(self, subject: str, trace: list) -> dict:
        trace.append(f"[LOOKUP] Tìm definition của '{subject}'")
        facts = self.knowledge.query(subject, 'is_a')
        if facts:
            objects = [f['object'] for f in facts[:3]]
            conf = facts[0]['confidence']
            trace.append(f"[FOUND] is_a: {objects} (conf={conf:.2f})")
            return {
                'answer': f"{subject.capitalize()} là {', '.join(objects)}.",
                'confidence': conf,
                'hallucination_risk': 'LOW',
                'found_at_level': 0
            }
        # Climb
        return self._climb_ontology_for(subject, 'is_a', trace)

    def _answer_object(self, subject: str, verb: str, trace: list) -> dict:
        trace.append(f"[LOOKUP] Tìm '{subject}' + '{verb}' trực tiếp")
        facts = self.knowledge.query(subject, verb)
        if facts:
            objects = [f['object'] for f in facts[:4]]
            conf = facts[0]['confidence']
            trace.append(f"[FOUND] {verb}: {objects} (conf={conf:.2f})")
            return {
                'answer': f"{subject.capitalize()} {verb} {', '.join(objects)}.",
                'confidence': conf,
                'hallucination_risk': 'LOW',
                'found_at_level': 0
            }
        trace.append(f"[MISS] Không tìm thấy trực tiếp — leo ontology")
        return self._climb_for_predicate(subject, verb, trace)

    def _answer_property(self, subject: str, prop: str, trace: list) -> dict:
        trace.append(f"[LOOKUP] Tìm property '{prop}' của '{subject}'")

        # Tìm trực tiếp
        facts = self.knowledge.query(subject, 'property')
        for f in facts:
            if prop.lower() in f['object'].lower():
                trace.append(f"[FOUND] property trực tiếp (conf={f['confidence']:.2f})")
                hedge = " (theo quan điểm phổ biến)" if ReasoningKB.is_subjective(prop) else ""
                return {
                    'answer': f"{subject.capitalize()} {prop}{hedge}.",
                    'confidence': f['confidence'],
                    'hallucination_risk': 'LOW',
                    'found_at_level': 0
                }

        # Thử synonym của prop — VD: hỏi "đáng yêu", KB có "dễ thương"
        syns = self.knowledge.get_synonyms(prop)
        for syn in syns:
            for f in facts:
                if syn in f['object'].lower():
                    trace.append(f"[SYNONYM] '{prop}' ≈ '{syn}' → tìm thấy (conf={f['confidence']:.2f})")
                    hedge = " (theo quan điểm phổ biến)" if ReasoningKB.is_subjective(prop) else ""
                    return {
                        'answer': f"{subject.capitalize()} {prop}{hedge} (tương tự {syn}).",
                        'confidence': f['confidence'] * 0.95,
                        'hallucination_risk': 'LOW',
                        'found_at_level': 0
                    }

        trace.append(f"[MISS] Không có trực tiếp — leo ontology")
        return self._climb_property(subject, prop, trace)

    def _answer_location(self, subject: str, trace: list) -> dict:
        trace.append(f"[LOOKUP] Tìm location của '{subject}'")
        facts = self.knowledge.query(subject, 'lives_in')
        if not facts:
            facts = self.knowledge.query(subject, 'is_a')
        if facts:
            obj = facts[0]['object']
            conf = facts[0]['confidence']
            trace.append(f"[FOUND] location: {obj} (conf={conf:.2f})")
            return {
                'answer': f"{subject.capitalize()} ở/sống {obj}.",
                'confidence': conf,
                'hallucination_risk': 'LOW',
                'found_at_level': 0
            }
        return self._unknown(subject, trace)

    def _answer_general(self, subject: str, trace: list) -> dict:
        trace.append(f"[LOOKUP] Tìm general knowledge về '{subject}'")
        facts = self.knowledge.query(subject)[:5]
        if facts:
            summary = '; '.join([f"{f['predicate']} {f['object']}" for f in facts[:3]])
            trace.append(f"[FOUND] {len(facts)} facts")
            return {
                'answer': f"{subject.capitalize()}: {summary}.",
                'confidence': facts[0]['confidence'],
                'hallucination_risk': 'LOW',
                'found_at_level': 0
            }
        return self._unknown(subject, trace)

    def _climb_ontology_for(self, subject: str, predicate: str,
                             trace: list, level: int = 0,
                             base_conf: float = 1.0) -> dict:
        if level >= ReasoningKB.MAX_ONTOLOGY_LEVELS:
            trace.append(f"[STOP] Leo hết {level} tầng — không tìm thấy")
            return self._unknown(subject, trace)

        parents = self.vocab.get_parents(subject)
        if not parents:
            trace.append(f"[STOP] '{subject}' không có parent trong ontology")
            return self._unknown(subject, trace)

        for parent in parents[:3]:
            decayed_conf = base_conf * (ReasoningKB.CONFIDENCE_DECAY ** (level + 1))
            trace.append(f"[CLIMB L{level+1}] {subject} → {parent} (conf decay → {decayed_conf:.2f})")

            facts = self.knowledge.query(parent, predicate)
            if facts:
                objects = [f['object'] for f in facts[:3]]
                final_conf = min(facts[0]['confidence'], decayed_conf)
                trace.append(f"[FOUND L{level+1}] Kế thừa từ '{parent}': {objects} (conf={final_conf:.2f})")
                if final_conf < ReasoningKB.CONFIDENCE_THRESHOLD:
                    trace.append(f"[WARN] Confidence {final_conf:.2f} < threshold {ReasoningKB.CONFIDENCE_THRESHOLD} — không đủ tin cậy")
                    return self._unknown(subject, trace)
                return {
                    'answer': f"{subject.capitalize()} là {', '.join(objects)} (kế thừa từ '{parent}').",
                    'confidence': final_conf,
                    'hallucination_risk': 'LOW',
                    'found_at_level': level + 1,
                    'inherited_from': parent
                }

        # Không tìm thấy ở level này, leo tiếp
        for parent in parents[:1]:
            result = self._climb_ontology_for(parent, predicate, trace, level + 1, base_conf)
            if result.get('confidence', 0) > ReasoningKB.CONFIDENCE_THRESHOLD:
                return result

        return self._unknown(subject, trace)

    def _climb_for_predicate(self, subject: str, predicate: str,
                              trace: list, level: int = 0,
                              base_conf: float = 1.0) -> dict:
        if level >= ReasoningKB.MAX_ONTOLOGY_LEVELS:
            return self._unknown(subject, trace)

        parents = self.vocab.get_parents(subject)
        for parent in parents[:3]:
            decayed_conf = base_conf * (ReasoningKB.CONFIDENCE_DECAY ** (level + 1))
            trace.append(f"[CLIMB L{level+1}] {subject} → {parent}")
            facts = self.knowledge.query(parent, predicate)
            if facts:
                objects = [f['object'] for f in facts[:4]]
                final_conf = min(facts[0]['confidence'], decayed_conf)
                trace.append(f"[FOUND L{level+1}] Kế thừa '{predicate}' từ '{parent}' (conf={final_conf:.2f})")
                if final_conf < ReasoningKB.CONFIDENCE_THRESHOLD:
                    return self._unknown(subject, trace)
                return {
                    'answer': f"{subject.capitalize()} {predicate} {', '.join(objects)} (dựa trên '{parent}').",
                    'confidence': final_conf,
                    'hallucination_risk': 'LOW',
                    'found_at_level': level + 1,
                    'inherited_from': parent
                }
        if parents:
            return self._climb_for_predicate(parents[0], predicate, trace, level + 1, base_conf)
        return self._unknown(subject, trace)

    def _climb_property(self, subject: str, prop: str,
                         trace: list, level: int = 0,
                         base_conf: float = 1.0) -> dict:
        if level >= ReasoningKB.MAX_ONTOLOGY_LEVELS:
            return self._unknown(subject, trace)

        parents = self.vocab.get_parents(subject)
        for parent in parents[:3]:
            decayed_conf = base_conf * (ReasoningKB.CONFIDENCE_DECAY ** (level + 1))
            trace.append(f"[CLIMB L{level+1}] {subject} → {parent} (conf → {decayed_conf:.2f})")
            facts = self.knowledge.query(parent, 'property')
            for f in facts:
                if prop.lower() in f['object'].lower():
                    final_conf = min(f['confidence'], decayed_conf)
                    trace.append(f"[FOUND L{level+1}] Property '{prop}' kế thừa từ '{parent}' (conf={final_conf:.2f})")
                    if final_conf < ReasoningKB.CONFIDENCE_THRESHOLD:
                        return self._unknown(subject, trace)
                    hedge = " (theo quan điểm phổ biến)" if ReasoningKB.is_subjective(prop) else ""
                    return {
                        'answer': f"{subject.capitalize()} {prop}{hedge} (dựa trên '{parent}').",
                        'confidence': final_conf,
                        'hallucination_risk': 'LOW',
                        'found_at_level': level + 1,
                        'inherited_from': parent
                    }
        if parents:
            return self._climb_property(parents[0], prop, trace, level + 1, base_conf)
        return self._unknown(subject, trace)

    def _unknown(self, subject: str, trace: list) -> dict:
        trace.append(f"[UNKNOWN] Không đủ thông tin để trả lời")
        return {
            'answer': f"Tôi chưa có đủ thông tin về '{subject}' để trả lời câu hỏi này.",
            'confidence': 0.0,
            'hallucination_risk': 'NONE',
            'found_at_level': -1
        }

    def _fallback_parse(self, query: str) -> dict:
        words = query.replace('?', '').strip().split()
        subject = ' '.join(words[:3]) if words else query
        return {'intent': 'general', 'subject': subject}
