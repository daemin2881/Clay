"""과거 행동의 계층(트리) 탐색: 영역에 영향을 준 과거 행동들을 '시간 구간 묶음'(노드)으로 나눠 묶음째 Genesis 에서 재생해 보고,
효과가 있는 묶음만 쪼개서 어떤 행동 때문인지 좁혀 간다. 대상(A, 하트, ...)과 영역에 의존하지 않는다. (Genesis/torch 불필요)

왜 이렇게 하나 (A 이력 80/190 행동에서 잰 것)
  - 영역에 영향을 주는 행동은 인과 폐포로 보면 190행동 중 143~172개(75~90%)라서 '영향 준 행동만' 따로 모아 재생하면 건너뛴 행동 때문에 control 이 기록과 안 맞는다.
    그래서 재생 길이는 줄일 수 없고(시작 행동 s0 부터 현재 K 까지 전체), 줄일 수 있는 것은 '재생하는 횟수'다.
  - 후보 하나를 재생하는 비용은 (K - s0 + 1) 행동이다(행동당 20~40초). 행동을 하나씩 재생하면 후보 수만큼 든다.
    같은 구간의 행동 여러 개를 '한 번에' 바꿔 한 번만 재생하면(그룹 반사실) 한 번에 여러 행동의 효과를 시험할 수 있고, 효과가 없는 묶음은 통째로 버릴 수 있다.
  - 사전 점수는 제한 최소제곱 가중치 w 와 영역 안 영향 rms 의 곱 |w|*영향(cf_compensate.dictionary). 선형 근사(기록된 dH 를 더하기)는 순위가 약해서(Genesis 재생 30쌍에서 Spearman +0.35,
    #75 깊이 x1.5 처럼 효과 +0.53 인 것을 근사는 -0.11 로 봤다) 사전 점수로는 쓰지 않는다. 가지치기는 근사가 아니라 Genesis 재생 결과로 한다.

구조
  노드 = 시간순으로 정렬된 행동 번호(1부터)들의 연속 구간. 사전 점수 prior = 구성 행동의 |w|*영향 합, 비용 cost = K - (가장 이른 행동) + 1.
  초기 노드: 관련 행동을 사전 점수 질량이 비슷하도록 --tree-init 개 구간으로 나눈 것.
  한 파동(wave): 대기 중인 노드를 prior/cost 큰 순으로 시간 예산 안에서 골라 재생. 재생 결과 효과(영역 점수 개선)가 기준(eps) 이상인 노드만 둘로 쪼개 다음 파동에 넣고,
  아닌 노드는 가지를 버린다. 노드가 행동 하나(잎)가 되면 그 행동이 원인 후보다.
"""
from __future__ import annotations


class Node:
    __slots__ = ("members", "prior", "depth", "nid", "parent")

    def __init__(self, members, prior, depth, nid, parent=None):
        self.members = list(members)          # 행동 번호(1부터), 시간순
        self.prior = float(prior)
        self.depth = int(depth)
        self.nid = nid
        self.parent = parent

    @property
    def s_min(self):
        return self.members[0]

    @property
    def is_leaf(self):
        return len(self.members) == 1

    def cost(self, K):
        """재생해야 하는 행동 수 (가장 이른 행동부터 현재 K 번째까지)."""
        return K - self.s_min + 1

    def __repr__(self):
        return f"Node(d{self.depth} #{self.members[0]}..#{self.members[-1]} n={len(self.members)} prior={self.prior:.2f})"


def _mass_split(members, pri, parts):
    """시간순 members 를 사전 점수 질량이 비슷한 parts 개의 연속 구간으로 나눈다(빈 구간은 버림)."""
    parts = max(1, min(int(parts), len(members)))
    w = [max(pri[m], 1e-12) for m in members]
    tot = sum(w)
    segs, cum = [[] for _ in range(parts)], 0.0
    for m, wi in zip(members, w):
        mid = cum + wi / 2.0
        segs[min(parts - 1, int(mid / (tot / parts)))].append(m)
        cum += wi
    return [s for s in segs if s]


def initial_nodes(rows, parts, K=None):
    """rows: [(행동 번호(1부터), w, 영역 안 영향)]. 반환: (노드 목록, prior 사전)."""
    if not rows:
        return [], {}
    pri = {int(t): abs(float(w)) * float(e) for t, w, e in rows}
    members = sorted(pri)
    nid = 0
    nodes = []
    for seg in _mass_split(members, pri, parts):
        nodes.append(Node(seg, sum(pri[m] for m in seg), 0, nid))
        nid += 1
    return nodes, pri


def children(node, pri, next_id, min_frac=0.0):
    """노드를 둘로 쪼갠다(사전 점수 질량 기준, 한쪽이 비면 개수 기준). 잎이면 [].
    min_frac > 0 이면 사전 점수가 노드의 min_frac 배 미만인 쪽은 재생하지 않는다(효과 있는 묶음 안에서 점수가 거의 없는 행동까지 긴 재생을 하지 않기 위해).
    둘 다 미만이면 둘 다 남긴다. 버린 쪽의 효과는 모른다(사전 점수가 낮을 뿐이다)."""
    if node.is_leaf:
        return []
    segs = _mass_split(node.members, pri, 2)
    if len(segs) < 2:
        h = len(node.members) // 2
        segs = [node.members[:h], node.members[h:]]
    kids = [Node(s, sum(pri[m] for m in s), node.depth + 1, next_id + i, node) for i, s in enumerate(segs)]
    if min_frac > 0:
        keep = [k for k in kids if k.prior >= min_frac * node.prior]
        kids = keep or kids
    return kids


def pick_wave(frontier, K, budget_actions, max_nodes=None, max_single=None):
    """대기 노드 중 prior/cost 가 큰 순으로 재생 예산(재생할 행동 수의 합) 안에서 고른다. 예산이 모자라면 건너뛴다(예산 초과 방지).
    max_single: 노드 하나의 재생은 워커 여러 개로 나눌 수 없어서 소요 시간이 (재생 행동 수 x 행동당 초)다. 남은 예산 시간 안에 못 끝나는 노드(cost > max_single)는 고르지 않는다.
    반환: (고른 노드, 남은 노드)."""
    order = sorted(frontier, key=lambda nd: -nd.prior / max(nd.cost(K), 1))
    chosen, rest, spent = [], [], 0
    for nd in order:
        c = nd.cost(K)
        if max_single is not None and c > max_single:
            rest.append(nd)
            continue
        if (max_nodes is None or len(chosen) < max_nodes) and spent + c <= budget_actions:
            chosen.append(nd)
            spent += c
        else:
            rest.append(nd)
    return chosen, rest


def variant_plan(node, w_of, vars_for):
    """노드의 각 행동에 w 의 부호 방향 변형 하나(첫 번째)를 붙인다. 반환: {행동 번호: (변형 이름, 길이배율, 깊이배율)}."""
    return {m: vars_for(w_of[m])[0] for m in node.members}


if __name__ == "__main__":      # 단위 시험
    rows = [(t, w, e) for t, w, e in [(2, .59, 2.6), (5, .05, .4), (8, .23, 2.6), (22, .29, 2.2), (29, .2, 3.6), (42, .21, 1.4), (55, .16, 2.2), (59, .12, 1.4), (62, .14, 1.9), (75, .17, 2.4)]]
    nodes, pri = initial_nodes(rows, 4)
    assert sum(len(n.members) for n in nodes) == len(rows) and all(n.members == sorted(n.members) for n in nodes)
    allm = [m for n in nodes for m in n.members]
    assert allm == sorted(allm), "구간은 연속(시간순)이어야 한다"
    K = 80
    ch, rest = pick_wave(nodes, K, budget_actions=100)
    assert sum(n.cost(K) for n in ch) <= 100 and len(ch) + len(rest) == len(nodes)
    ch2, rest2 = pick_wave(nodes, K, budget_actions=1000, max_single=30)
    assert all(n.cost(K) <= 30 for n in ch2) and len(ch2) + len(rest2) == len(nodes)
    # 쪼개기: 끝까지 쪼개면 잎은 행동 하나이고 합집합이 같다
    stack, leaves, nid = list(nodes), [], 100
    while stack:
        nd = stack.pop()
        kids = children(nd, pri, nid)
        nid += len(kids)
        if not kids:
            leaves.append(nd)
        stack += kids
    assert sorted(m for l in leaves for m in l.members) == sorted(pri) and all(l.is_leaf for l in leaves)
    # 한 행동이 질량을 독점해도 두 노드로 쪼개진다
    nd = Node([2, 5, 8], 3.0, 0, 0)
    kids = children(nd, {2: 100.0, 5: 0.01, 8: 0.01}, 1)
    assert len(kids) == 2 and sorted(m for k in kids for m in k.members) == [2, 5, 8]
    kids = children(nd, {2: 100.0, 5: 0.01, 8: 0.01}, 1, min_frac=0.05)
    assert len(kids) == 1 and kids[0].members == [2], kids       # 점수가 거의 없는 쪽은 재생하지 않는다
    print("cf_tree 단위 시험 통과:", [repr(n) for n in nodes])
