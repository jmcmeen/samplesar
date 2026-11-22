# The nesting strategies, formally

The Methods of the hull-strategies paper, and the specification the code is
checked against: each strategy below names the function that implements it,
and a disagreement between the two is a bug in one of them.

## Notation

**Sample.** A sample is a set of observations, each a species at a location.
Locations are projected to the plane (km): by default a Lambert azimuthal
equal-area projection of the WGS84 ellipsoid centred on the points' mean
direction (or on a given `centre`), or the sample's own projected coordinate
system (`Sample.xy`). Let

- $P = \{p_1, \dots, p_n\} \subset \mathbb{R}^2$ be the projections of the
  distinct (longitude, latitude) points, indexed in the order first seen;
- $S(p)$ be the set of species observed at point $p$.

For any subset $Q \subseteq P$:

- **Richness** $R(Q) = \left|\bigcup_{p \in Q} S(p)\right|$.
- **Hull** $H(Q) = \operatorname{conv}(Q)$, and **area** $A(Q)$ its area in
  km² (divided by the coordinate system's areal scale at the sample's centre
  when that system isn't equal-area).
- **Vertices** $V(Q) = \{p \in Q : p \notin \operatorname{conv}(Q \setminus
  \{p\})\}$: the points where the hull bends. A point on an edge between two
  vertices is not a vertex; in floating point, "on" means within
  $\delta(Q) = 10^{-9} \cdot \operatorname{diag}(H(Q))$ of the line through
  its neighbours on the hull, where $\operatorname{diag}$ is the diagonal of
  the hull's bounding box. If the points of $Q$ are collinear, $V(Q)$ is the
  two ends of the segment.
- **Boundary** $B(Q) = \{p \in Q : \operatorname{dist}(p, \partial H(Q)) \le
  \delta(Q)\}$: the vertices and any points on the hull's edges. If $H(Q)$
  isn't a polygon (fewer than three points, or collinear), $B(Q) = Q$.
- **Centre** $c(Q) = \frac{1}{|V(Q)|} \sum_{v \in V(Q)} v$, the centroid of
  the vertices, and $d_Q(p) = \lVert p - c(Q) \rVert$.
- **Holders** $h_Q(s) = |\{p \in Q : s \in S(p)\}|$, the number of points in
  $Q$ where species $s$ was observed, and the **loss** of removing $p$,
  $L_Q(p) = |\{s \in S(p) : h_Q(s) = 1\}|$, the species found only at $p$.
  Then $R(Q \setminus \{p\}) = R(Q) - L_Q(p)$.

**Nest.** A nest is a sequence $Q_0 \supset Q_1 \supset \dots \supset Q_T$
with $Q_0 = P$. Its species-area data are the pairs $(A(Q_t), R(Q_t))$ for
the steps with $A(Q_t) > 0$, which are what a SAR model is fitted to.

**Stopping.** Every strategy takes a minimum size $m \ge 1$ (`min_points`,
default 3) and needs $n \ge m$. The one-at-a-time strategies stop when
$|Q_t| = m$, so $T = n - m$; hull peeling stops before a step would leave
fewer than $m$ points.

**Ties.** The deterministic strategies pick one vertex by a score $f$.
Scores computed at real coordinates differ in their last bits even when
equal in exact arithmetic, so ties are taken within a relative tolerance
$\varepsilon = 10^{-9}$:

$$
\operatorname{top}_\varepsilon(f, C) = \{v \in C : f(v) \ge \max_C f - \varepsilon \max_C |f|\}.
$$

Unless a strategy says otherwise, its choice is
$\operatorname{top}_\varepsilon(f, V(Q_t))$, then among those
$\operatorname{top}_\varepsilon(d_{Q_t}, \cdot)$ (the vertex farthest from
the centre), then the lowest index. So the nest is a function of the sample
alone.

## One point at a time

Each of these sets $Q_{t+1} = Q_t \setminus \{r_t\}$ for a removed point
$r_t$:

| | Strategy | $r_t$ | Function |
| --- | --- | --- | --- |
| S1 | Random | uniform on $Q_t$ | `strategies.random` |
| S2 | Random hull vertex | uniform on $V(Q_t)$ | `strategies.hull_random` |
| S4 | Largest remaining area | maximises $A(Q_t \setminus \{v\})$ over $v \in V(Q_t)$ | `strategies.hull_area(keep="largest")` |
| S5 | Smallest remaining area | minimises $A(Q_t \setminus \{v\})$ | `strategies.hull_area(keep="smallest")` |
| S6 | Most remaining species | minimises $L_{Q_t}(v)$ | `strategies.hull_species(keep="most")` |
| S7 | Fewest remaining species | maximises $L_{Q_t}(v)$ | `strategies.hull_species(keep="fewest")` |
| S8 | Farthest from the centre | maximises $d_{Q_t}(v)$; ties by index only | `strategies.hull_distance(remove="farthest")` |
| S9 | Closest to the centre | minimises $d_{Q_t}(v)$; ties by index only | `strategies.hull_distance(remove="closest")` |

S1 and S2 are random (a seed makes them reproducible); S4 to S9 are
deterministic.

## A layer at a time

| | Strategy | Step | Function |
| --- | --- | --- | --- |
| S3 | Hull peeling | $Q_{t+1} = Q_t \setminus B(Q_t)$, while $\lvert Q_{t+1} \rvert \ge m$ | `strategies.hull_peel` |

S3 gives the sample's convex layers (onion peeling). It removes many points
per step, so its nest is short: $T$ is the number of layers peeled while at
least $m$ points remain inside, at most one fewer than the number of layers
(the innermost is never peeled). If not even the first layer can be peeled,
because fewer than $m$ points lie strictly inside the hull, it raises an
error rather than return a nest with nothing to fit.

## Properties

1. **Nested and monotone.** Every strategy gives $Q_{t+1} \subset Q_t$, so
   $A(Q_{t+1}) \le A(Q_t)$ and $R(Q_{t+1}) \le R(Q_t)$.
2. **The hull shrinks only when a vertex goes.** For S2 to S9 every removed
   point is on the boundary, so $H(Q_{t+1}) \subseteq H(Q_t)$ with the area
   falling at nearly every step. S1 removes interior points too, which leave
   the hull unchanged: its area stays near $A(P)$ until the last few points,
   and its species-area curve is close to vertical.
3. **Greedy, not optimal.** S4 to S7 choose the best single step; they don't
   find the order that keeps the most area (or species) over the whole
   nest. Optimising whole orders is a separate problem; any order can be
   evaluated as a nest with `strategies.from_order`.
4. **Exactness of S6 and S7.** Because $R(Q \setminus \{v\}) = R(Q) -
   L_Q(v)$, keeping $h_Q$ up to date gives each step's choice exactly,
   without recounting the sample.

## Cost

With $n$ points, $h$ hull vertices per step (typically $h \ll n$) and $k$
species per point:

| Strategies | Per step | Whole nest |
| --- | --- | --- |
| S1 | $O(1)$ | $O(n)$ |
| S2, S8, S9 | one hull, $O(n \log n)$, plus $O(h^2)$ at worst to drop corners rounding pushed off straight edges | $O(n^2 \log n)$ |
| S3 | one hull per layer | $O(n^2 \log n)$ at worst |
| S4, S5 | $h$ hulls, $O(h\, n \log n)$ | $O(h\, n^2 \log n)$ |
| S6, S7 | one hull and $O(hk)$ | $O(n^2 \log n)$ |

Measured on an 893-point sample (one laptop): S4 and S5 took 0.5 to 0.8 s,
every other strategy under 0.2 s.

## Open questions for the Methods

- **Co-located records.** Records at one spot differ by GPS jitter, so the
  last steps of S2 to S9 measure centimetre-scale hulls that aren't area.
  Merging points within a distance, or a minimum area, needs choosing and
  justifying (samplesar #46).
- **Pseudo-replication.** Steps share most of their points, so R², AICc and
  standard errors from one nest aren't independent evidence. Strategies
  should be compared on the same sample, and inference built on resampling
  (bootstrapped samples, or many seeds for S1 and S2).
- **Model bounds.** Near-vertical curves (S1, S9) drive the power model's $z$
  to its bound (samplesar #48).
- **Which vertex centre.** $c(Q)$ is the centroid of the vertices, which
  weights a densely sampled stretch of the hull more; the area centroid of
  $H(Q)$ is the alternative.
