# CLEAR: Constraint-Compatible Lateral Escape with Audited Rigid Motion

## 0. Scope and central claim

This document defines the theory implemented in `clear_nav/`.  It is intentionally
independent of the earlier learned rule, MGR reconstruction, ORCA adapter, and
bounded-dither experiments.

The central idea is:

> Deadlock-breaking motion should be generated inside the one-sided tangent
> cone of the current safety constraints *before* the safety filter is applied.

This is stronger than adding a vortex and hoping that a CBF preserves it.  The
circulation is first made compatible with every nearby robot and obstacle
constraint.  A final CBF projection then supplies forward invariance.  The
result gives four distinct objects:

1. a deterministic geometric escape mechanism;
2. an exact continuous-time safety certificate; and
3. a state-wise, numerically auditable certificate excluding static deadlock;
   and
4. a controller-specific finite-exit theorem for a straight two-sided bridge.

The third item does not exclude periodic orbits.  The fourth excludes them only
inside its stated bridge cell.  Global arrival in arbitrary clutter remains
conditional and is stated separately.

---

## 1. Model and safe set

The geometric layer is first written for an abstract controlled point
\(x_i\) with first-order dynamics

\[
\dot x_i=u_i,\qquad x_i,u_i\in\mathbb R^2,\qquad \|u_i\|\le v_{\max}.
\tag{1}
\]

In the reported controller this point is the look-ahead point of a bounded
unicycle, not a separate physical single-integrator robot.
Corollary 2a gives the native input projection and the transfer from virtual
clearance to physical-center clearance.

Let \(d_{\rm pair}>0\) be the certified centre-to-centre separation.  For
\(i<j\), define

\[
p_{ij}=x_i-x_j,\quad r_{ij}=\|p_{ij}\|,\quad
n_{ij}=p_{ij}/r_{ij},\quad
h_{ij}=r_{ij}-d_{\rm pair}.
\tag{2}
\]

Each static obstacle \(O_q\) is a closed convex set.  With robot clearance
\(d_{\rm obs}\),

\[
h_{iq}(x_i)=\operatorname{dist}(x_i,O_q)-d_{\rm obs}.
\tag{3}
\]

Workspace walls use affine barriers of the same sign convention.  The safe set
is

\[
\mathcal S=\{x:h_e(x)\ge0\text{ for every pair, obstacle, and wall edge }e\}.
\tag{4}
\]

Outside an obstacle, projection onto a closed convex set is unique; hence the
distance gradient is the outward unit normal.  Every barrier row is denoted

\[
a_e(x)^\top=\nabla_x h_e(x)^\top.
\tag{5}
\]

For a pair, \(a_{ij}\) has \(n_{ij}\) in robot \(i\)'s block and
\(-n_{ij}\) in robot \(j\)'s block.  For a static boundary it has only one
nonzero robot block.

---

## 2. Goal field and raw circulation

Let \(q_i\) be robot \(i\)'s goal.  The bounded goal field is

\[
g_i(x_i,q_i)
=v_{\max}\tanh(k_g\|q_i-x_i\|/\rho_g)
\frac{q_i-x_i}{\max(\|q_i-x_i\|,\varepsilon)}.
\tag{6}
\]

Let \(J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}\), and let the global handedness
\(\chi\in\{-1,+1\}\) be a physical convention.  It is not a robot ID or a
random variable.

### 2.1 Robot--robot term

For every sensed pair, choose a symmetric scalar
\(\beta_{ij}=\beta_{ji}\ge0\) from distance and predicted closing speed.  The
code uses a compact \(C^1\) distance gate and a nonnegative closing-speed gate.
Their normalized product gives \(0\le\beta_{ij}\le1\).
Define

\[
c_i^{ij}=\chi\beta_{ij}Jn_{ij},\qquad
c_j^{ij}=-\chi\beta_{ij}Jn_{ij}.
\tag{7}
\]

### 2.2 Robot--boundary term

For the outward normal \(n_{iq}=\nabla h_{iq}\), choose
\(\beta_{iq}\ge0\) and define

\[
c_i^{iq}=\chi\beta_{iq}Jn_{iq}.
\tag{8}
\]

With boundary gain \(k_b\), the implementation satisfies
\(0\le\beta_{iq}\le k_b/\kappa_0\).

The raw circulation \(c(x)\in\mathbb R^{2N}\) is the sum of (7), (8), and wall
terms.  This is a unified law: a robot passes another robot and follows an
obstacle boundary using the same oriented \(90^\circ\) generator.

### Proposition 1 (reciprocity, tangency, and pair moment)

For every robot pair,

\[
c_i^{ij}+c_j^{ij}=0,\qquad
n_{ij}^{\mathsf T}(c_i^{ij}-c_j^{ij})=0.
\tag{9}
\]

Moreover, with centroid-relative coordinates
\(y_i=x_i-\frac1N\sum_kx_k\), its kinematic moment is

\[
y_i\times c_i^{ij}+y_j\times c_j^{ij}
=\chi\beta_{ij}r_{ij}.
\tag{10}
\]

For every static boundary term,

\[
n_{iq}^{\mathsf T}c_i^{iq}=0.
\tag{11}
\]

**Proof.**  Since \(n^\top Jn=0\), the relative pair contribution in (9) is
tangent to the constant-distance circle.  Reciprocity follows directly from
(7).  Also

\[
(y_i-y_j)\times(\chi\beta_{ij}Jn_{ij})
=p_{ij}\times(\chi\beta_{ij}Jn_{ij})
=\chi\beta_{ij}r_{ij}.
\]

This proves (10).  Equation (11) again follows from
\(n^\top Jn=0\). \(\square\)

---

## 3. The one-sided tangent-cone layer

A raw boundary term can be tangent to its own obstacle and still point inward
with respect to another nearby obstacle or robot.  CLEAR removes this cross-edge
conflict explicitly.

For a band \(\delta_T>0\), collect all barriers satisfying
\(h_e(x)\le\delta_T\) into the matrix \(A_T(x)\).  Define the linearized
one-sided tangent cone

\[
\mathcal K_T(x)=\{z\in\mathbb R^{2N}:A_T(x)z\ge0\}.
\tag{12}
\]

The compatible base circulation is the Euclidean cone projection

\[
z_{\rm base}(x)=\Pi_{\mathcal K_T(x)}c(x)
=\arg\min_{\zeta\in\mathcal K_T(x)}
\frac12\|\zeta-c(x)\|^2.
\tag{13}
\]

Unlike a nullspace projection, (13) permits motion that *increases* a binding
clearance.  This matters at corners and rigid contact graphs, where the exact
nullspace may contain no useful translational escape direction.

### Proposition 2 (barrier compatibility and least modification)

The projection in (13) exists and is unique.  It satisfies

\[
A_Tz_{\rm base}\ge0,\qquad
\langle c-z_{\rm base},z_{\rm base}\rangle=0,\qquad
\|z_{\rm base}\|\le\|c\|.
\tag{14}
\]

It is the smallest Euclidean modification of the raw circulation that is
first-order non-inward for every constraint in the tangent band.

**Proof.**  \(\mathcal K_T\) is a nonempty closed convex cone because it is an
intersection of homogeneous half-spaces and contains the origin.  Existence and
uniqueness follow from the projection theorem.  Moreau's cone decomposition
gives \(c=z_{\rm base}+(c-z_{\rm base})\), orthogonality of the two terms, and
the norm bound.
\(\square\)

### 3.1 Audited rigid component extension

Let \(C\) be a connected component of the structural pair graph and let
\(\mathcal B_C\) be its incident structural boundary contacts.  For a unit
direction \(d_C\), define the rigid vector

\[
[R_C(d_C)]_i=
\begin{cases}
d_C,&i\in C,\\
0,&i\notin C.
\end{cases}
\tag{14a}
\]

Every internal pair row annihilates this vector.  CLEAR forms candidates from
the aggregate guide, incident boundary normals, and their two quarter turns,
retains only

\[
n_{ib}^{\mathsf T}d_C\ge0
\quad\forall(i,b)\in\mathcal B_C,
\tag{14b}
\]

and selects the feasible candidate maximizing
\((\sum_{i\in C}g_i)^{\mathsf T}d_C\).  The complete rigid vector is then
audited against every row of \(A_T\).  A feasible direction is stored under
the component key while the same key remains active and the stored direction
continues to satisfy (14b).

With nonnegative component weights \(\rho_C\), the complete passage field is

\[
z=z_{\rm base}+\sum_C\rho_CR_C(d_C).
\tag{14c}
\]

Every accepted term lies in \(\mathcal K_T\), and this set is a convex cone.
Therefore \(z\in\mathcal K_T\).  A common rigid velocity also preserves every
internal pair clearance because
\(n_{ij}^{\mathsf T}(d_C-d_C)=0\).

The standard controller uses a fixed gain

\[
w(x)=g(x)+\kappa_0z(x),\qquad \kappa_0>0.
\tag{15}
\]

The nonzero pair-circulation edges define connected components
\(\mathcal C(x)\); a robot interacting only with a boundary forms a singleton
component.  These components are used to audit deadlock witnesses.  An
adaptive certificate-enforcing gain is retained as an ablation, not as the
standard controller.

---

## 4. Final CBF and actuator projection

The CBF half-space set is

\[
\mathcal H_{\rm CBF}(x)=
\left\{
u:
a_e(x)^\top u\ge-\gamma h_e(x)\ \forall e
\right\}.
\tag{16}
\]

First compute

\[
\bar u(x)=\Pi_{\mathcal H_{\rm CBF}(x)}w(x).
\tag{17a}
\]

The actuator bound is imposed by one common positive scale:

\[
\alpha(x)=\min\left\{1,
\frac{v_{\max}}{\max_i\|\bar u_i(x)\|}\right\},\qquad
u^*(x)=\alpha(x)\bar u(x).
\tag{17b}
\]

Common scaling is structural.  Independent per-robot clipping can violate a
relative pair constraint.  At a safe state every CBF right-hand side is
nonpositive, so multiplication by \(\alpha\in(0,1]\) preserves every CBF
half-space.  The implementation solves (13) and (17a) by deterministic dual
coordinate projection and applies (17b) exactly.

### Theorem 1 (feasibility and continuous-time safety)

For every \(x\in\mathcal S\), \(\mathcal H_{\rm CBF}(x)\) is nonempty.  If
(17a) is solved exactly and the closed-loop solution exists, then
\(\mathcal S\) is forward invariant.

**Proof.**  At a safe state, \(u=0\) satisfies
\(0\ge-\gamma h_e(x)\) for every edge, proving feasibility.  Projection gives
\(a_e^\top\bar u\ge-\gamma h_e\).  If \(a_e^\top\bar u<0\), multiplying it by
\(\alpha\le1\) makes it less negative; if it is nonnegative, it remains
nonnegative.  Therefore under (17b),

\[
\dot h_e=a_e^\top u^*\ge-\gamma h_e.
\]

The comparison lemma gives
\(h_e(t)\ge e^{-\gamma t}h_e(0)\ge0\). \(\square\)

### Theorem 2 (zero-order-hold safety for the implemented primitives)

Assume all obstacle primitives are convex, the command is constant on a sample
\([t_k,t_k+\Delta t]\), (16) is satisfied at \(t_k\), and
\(\gamma\Delta t\le1\).  Then every pair, convex-obstacle, and affine-wall
barrier remains nonnegative throughout that sample.

**Proof.**  Each pair barrier is a Euclidean norm minus a constant in relative
coordinates, hence convex.  For convex \(O_q\),
\(\operatorname{dist}(\cdot,O_q)\) is convex, and wall barriers are affine.
For any \(\tau\in[0,\Delta t]\), the first-order inequality for a convex
function gives

\[
h_e(x+\tau u^*)\ge h_e(x)+\tau a_e^\top u^*
\ge(1-\gamma\tau)h_e(x)\ge0.
\]

\(\square\)

This theorem is why the code may omit a far-away constraint only when bounded
actuation makes its CBF inequality automatically true:
\(\gamma h_{ij}\ge2v_{\max}\) for a pair and
\(\gamma h_{iq}\ge v_{\max}\) for a static boundary.

### Corollary 2a (safety-preserving unicycle realization)

For a unicycle center \(x_i\), heading \(\theta_i\), and look-ahead distance
\(\ell>0\), define

\[
y_i=x_i+\ell e_i,\qquad
e_i=[\cos\theta_i,\sin\theta_i]^\top .
\tag{17c}
\]

Run CLEAR's nominal field on \(y\) with pair clearance
\(d_{\min}+2\ell\) and obstacle clearance enlarged by \(\ell\).  Define
\(z_i=[v_i,\ell\omega_i]^\top\) and
\(B_i=[e_i,Je_i]\), so that \(\dot y_i=B_i z_i\).  Stack these blocks in
\(B(\theta)\), and solve the final projection directly in bounded unicycle
coordinates:

\[
\begin{aligned}
z^*=\arg\min_z\;&\frac12\|z-B(\theta)^\top u^{\rm nom}\|^2,\\
\text{s.t. }&
A_y B(\theta)z\ge b_y,\\
&q_C^\top B(\theta)z\ge m v_C^{\rm uni}
\quad\text{for certified bridge components},\\
&-v_{\max}\le v_i\le v_{\max},\\
&-\ell\omega_{\max}\le z_{i,2}\le\ell\omega_{\max}.
\end{aligned}
\tag{17d}
\]

Here \(A_y\dot y\ge b_y\) contains the pair and boundary CBF rows.  Since
\(B_i\) is orthogonal, the objective is exactly the squared virtual-velocity
error.  At every safe state \(b_y\le0\), so \(z=0\) is feasible; the actuator
bounds therefore do not compromise feasibility.  The resulting
\(\dot y=B(\theta)z^*\) satisfies every included virtual CBF row directly.
Far omitted rows are automatically satisfied when the geometric cutoff uses
\(\bar v_y=\sqrt{v_{\max}^2+(\ell\omega_{\max})^2}\).  Moreover,

\[
\|x_i-x_j\|\ge\|y_i-y_j\|-2\ell\ge d_{\min}
\tag{17e}
\]

and the 1-Lipschitz property of distance to a closed obstacle set gives

\[
\operatorname{dist}(x_i,O)
\ge\operatorname{dist}(y_i,O)-\ell
\ge d_{\rm obs}.
\tag{17f}
\]

Thus virtual-point safety transfers directly to the physical centers.  This
is stronger than simulating an unconstrained heading tracker after CLEAR:
the nonholonomic input limits and the safety rows are enforced in the same
projection.  Independent post-projection clipping is not used because it can
invalidate relative pair CBF rows.
The optional certified bridge row is defined below.  It is appended only
after a closed-form rigid witness passes every current CBF and input-box
residual, so it does not require a second optimization.

---

## 5. A state-wise certificate excluding static deadlock

Define the global **tangent witness margin**

\[
\Delta_T(x)=\langle w(x),z(x)\rangle
=\langle g(x),z(x)\rangle+\kappa_0\|z(x)\|^2.
\tag{18}
\]

The implementation computes the corresponding margin for every nonzero
circulation-connected component and logs their minimum.  Positivity of all
component margins implies \(\Delta_T>0\).  The certificate is asserted only at
states where the relevant margin is positive.
For such a component,
\[
\Delta_{T,C}=\langle g_C,z_C\rangle+\kappa_C\|z_C\|^2.
\]

### Theorem 3 (feasible-direction no-static-deadlock certificate)

Let \(x\in\mathcal S\).  Suppose the full tangent audit is exact,
\(z(x)\ne0\), and

\[
\Delta_T(x)>0.
\tag{19}
\]

Then the executed velocity from (17b) is nonzero; hence \(x\) is not a static
closed-loop equilibrium.

**Proof.**  The full tangent audit gives \(a_e^\top z\ge0\) for every
constraint in the tangent band.  Every constraint outside the band has
\(h_e>0\).
Therefore a sufficiently small \(\epsilon>0\) satisfies

\[
a_e^\top(\epsilon z)\ge-\gamma h_e
\]

for every edge.  Thus
\(\epsilon z\in\mathcal H_{\rm CBF}(x)\).

Assume for contradiction that the projection in (17a) gives \(\bar u=0\).  The
variational inequality for Euclidean projection states

\[
\langle w-\bar u,v-\bar u\rangle\le0
\quad\text{for every }v\in\mathcal H_{\rm CBF}(x).
\]

Choose \(v=\epsilon z\).  Then
\(\epsilon\langle w,z\rangle\le0\), contradicting (19).  Therefore
\(\bar u\ne0\), and the positive scale in (17b) gives \(u^*\ne0\).
\(\square\)

### Corollary 3.1 (optional certificate-enforcing gain)

For a nonzero component, replacing \(\kappa_0\) by

\[
\kappa_C(x)=\max\left\{
\kappa_0,\,
-\frac{\langle g_C,z_C\rangle}{\|z_C\|^2}+\mu
\right\}
\tag{20a}
\]

yields

\[
\Delta_{T,C}\ge\mu\|z_C\|^2>0.
\tag{20b}
\]

This excludes exact fixed-gain cancellation without noise.  It is implemented
by `adaptive_certificate_gain=True`, but is not the default: the first
\(N=20\) regression made the witness positive at almost every congested step
while reducing Free and Swap arrival to \(40\%\) and \(60\%\), respectively.
This is direct evidence that static-deadlock exclusion is not a liveness
guarantee.

### Robust numerical version

If the implemented projection has velocity error \(\eta\), a conservative
certificate is

\[
\Delta_T(x)>\|z(x)\|\,\|\eta\|.
\tag{21}
\]

The reference code takes a stricter operational position: if the tangent
projection fails its residual tolerance, it replaces \(z\) by zero and records
a `tangent_fallback_step`.  A nonconverged final CBF step is also recorded and
is never counted as certified mission success.

---

## 6. Symmetry structure

Let \(P\) permute robot labels, \(R\in SO(2)\), and \(t\in\mathbb R^2\).
Transform robots, goals, and the whole environment by

\[
x'_i=Rx_{P(i)}+t,\qquad q'_i=Rq_{P(i)}+t.
\tag{22}
\]

### Theorem 4 (permutation and \(SE(2)\) equivariance)

If every scalar gate depends only on invariant distances and inner products,
then

\[
u^*(x',q',O')=(P\otimes R)u^*(x,q,O).
\tag{23}
\]

For a reflection \(Q\in O(2)\setminus SO(2)\),

\[
u^*_{-\chi}(Qx,Qq,QO)=Q\,u^*_{\chi}(x,q,O).
\tag{24}
\]

**Proof.**  Distances and closing-speed inner products are invariant.  Pair and
boundary normals transform by \(R\), while \(JR=RJ\) for \(R\in SO(2)\).
Euclidean projection commutes with orthogonal transformations and
permutations, so both projection layers preserve the transformation.  For a
reflection, \(JQ=-QJ\), which is compensated exactly by
\(\chi\mapsto-\chi\). \(\square\)

The fixed handedness is therefore precisely the minimal broken reflection
symmetry.  No label-dependent priority or random perturbation is hidden in the
law.

The current Python reference assembles each projection centrally.  Its
constraint matrix decomposes across connected components of the local
constraint graph, but a genuinely distributed numerical solver is future work;
the present code should not be described as a proved communication-free
implementation.

The implementation uses a memoryless Dijkstra cost field as its obstacle-aware
guide.
The safety theorems apply because the guide only supplies the reference in
(6); the final projection remains responsible for constraint enforcement.

---

## 7. Controller-specific finite exit in a straight bridge

The next result covers a restricted but useful class in which progress is
derived from the actual candidate rule, token logic, gates, and final CBF
projection.  It does not assume a measured descent rate or a retained
component fraction.

Choose orthogonal unit vectors \(n,d\) with \(d=Jn\).  For a component \(C\)
of size \(m\ge2\), define

\[
q_C=R_C(d),\qquad
s_C(x)=\frac1m\sum_{i\in C}d^\top x_i.
\tag{25}
\]

### Definition 1 (straight two-sided bridge cell)

A sampled congestion stratum \(\mathcal P(C,n,d,L)\) is a straight two-sided
bridge cell if, until it is exited:

1. \(C\) is one structural pair component, no sensed or predictive pair edge
   joins \(C\) to another component, and every sensed or predictive boundary
   normal incident to \(C\) belongs to \(\{n,-n\}\);
2. both signs occur among the structural boundary contacts;
3. \(g_i=a_i d\) with \(a_i\ge g_\parallel>0\) for every \(i\in C\);
4. at least one component pair and one incident boundary feature satisfy
   \(h_p\le\bar h_p<r_{Cp}\) and
   \(h_b\le\bar h_b<r_{Cb}\), respectively; and
5. \(s_C<s_{\rm out}=s_C(t_0)+L\).

The event starts at a token-creation sample.  Loss of any item, including a
change in component key, is by definition exit from this congestion stratum.
This is a geometric domain definition, not an assumed token duration.

Let

\[
\underline\sigma_p=\sigma(\bar h_p;r_{Cp})>0,\qquad
\underline\sigma_b=\sigma(\bar h_b;r_{Cb})>0.
\tag{26}
\]

Let \(k_C>0\) be the component gain and \(k_b\) the boundary-circulation gain.
The implemented component weight is
\(\rho_C=(k_C/\kappa_0)\sigma_p\sigma_b\), using the maximum active
component pair and boundary gates.
If at most \(E_p\) raw pair terms and \(E_b\) raw boundary terms are sensed in
the cell, set

\[
\begin{aligned}
W_{\mathcal P}
&=\sqrt N v_{\max}+\sqrt2\kappa_0E_p+k_bE_b+\sqrt N k_C,\\
\lambda_{\mathcal P}
&=\min\{1,v_{\max}/W_{\mathcal P}\},\\
v_{\mathcal P}
&=\lambda_{\mathcal P}
 (g_\parallel+k_C\underline\sigma_p\underline\sigma_b)>0.
\end{aligned}
\tag{27}
\]

### Theorem 5 (finite exit of a straight bridge cell)

Consider sampled CLEAR with period \(\Delta t\), fixed passage gain
\(\kappa_0\), progress-aligned boundary orientation, component hysteresis
enabled, and \(k_C>0\).  Starting from a token-creation sample in
\(\mathcal P(C,n,d,L)\), the implemented candidate rule selects \(d\), the
stored token cannot flip while the trajectory remains in the cell, and every
complete held sample that stays in the cell satisfies

\[
s_C(t_{k+1})-s_C(t_k)\ge\Delta t\,v_{\mathcal P}.
\tag{28}
\]

Consequently, the stratum is exited in at most

\[
K_{\rm exit}
=\left\lceil\frac{L}{\Delta t\,v_{\mathcal P}}\right\rceil
\tag{29}
\]

control periods and contains no sampled periodic orbit.  If
\(\gamma\Delta t\le1\), this exit is safe under the modeled zero-order-hold
dynamics.

**Proof.**  Because both \(n\) and \(-n\) occur, the only unit candidates
feasible for all incident boundary rows are \(d\) and \(-d\).  Their aggregate
guide scores have opposite signs:

\[
\left(\sum_{i\in C}g_i\right)^\top d
=\sum_{i\in C}a_i>0.
\tag{30}
\]

The implemented maximizer therefore selects \(d\).  At later samples,
\((\pm n)^\top d=0\), so the stored token passes its feasibility test and
cannot flip.  If the component key or bridge disappears, the stratum has
already ended.

The rigid vector \(q_C\) is orthogonal to every structural and predictive row:
an internal pair row annihilates common translation, an incident boundary row
has normal \(\pm n\), and every other row has support outside \(C\).  Projection
onto the structural cone therefore preserves the \(q_C\) component.  Internal
pair circulation is reciprocal and sums to zero, while progress-aligned
boundary circulation has nonnegative \(d\)-component.  Hence
\(q_C^\top z_{\rm base}\ge0\).

The two component gates are at least
\(\underline\sigma_p,\underline\sigma_b\).  After the outer gain
\(\kappa_0\) is applied, the rigid mode contributes at least
\(m k_C\underline\sigma_p\underline\sigma_b\) along \(q_C\); the guides
contribute at least \(m g_\parallel\).  Thus

\[
q_C^\top w
\ge m(g_\parallel+k_C\underline\sigma_p\underline\sigma_b).
\]

The final CBF feasible set is translation invariant along \(q_C\), so its
Euclidean projection preserves this component exactly:
\(q_C^\top\bar u=q_C^\top w\).
The gate bounds, \(\|z_{\rm base}\|\le\|c\|\), and disjoint component supports
give \(\|w\|\le W_{\mathcal P}\).  Since zero is CBF-feasible,
nonexpansiveness of projection gives \(\|\bar u\|\le W_{\mathcal P}\), hence
the common actuator scale is at least \(\lambda_{\mathcal P}\).
Therefore \(\dot s_C\ge v_{\mathcal P}\) throughout every held sample.
Integrating and summing proves (28)--(29); strict increase excludes a sampled
periodic orbit.  Theorem 2 supplies inter-sample safety. \(\square\)

This theorem is controller-specific but deliberately narrow.  It does not
cover corners, curved boundaries, changing look-ahead directions, or
cross-component encounters, nor does it exclude later re-entry into a
different congestion stratum.

### Corollary 5a (native bounded-input finite exit)

Let \(0<\eta<1\) and define

\[
G_{\mathcal P}
=g_\parallel+k_C\underline\sigma_p\underline\sigma_b,\qquad
v_{\mathcal P}^{\rm uni}
=\eta\min\{G_{\mathcal P},v_{\max},\ell\omega_{\max}\}>0.
\tag{30a}
\]

For each component certified to remain in the straight bridge cell, append
the single row

\[
q_C^\top B(\theta)z\ge m v_{\mathcal P}^{\rm uni}
\tag{30b}
\]

to (17d).  This augmented QP is feasible for every heading.  Indeed, choose
\(\bar v=\min\{G_{\mathcal P},v_{\max},\ell\omega_{\max}\}\), any
\(v_f\in[v_{\mathcal P}^{\rm uni},\bar v]\), and the rigid candidate
\(z_f=B(\theta)^\top v_fq_C\).  Every internal pair row annihilates this
common translation, incident boundary normals are orthogonal to \(d\), and
all other rows have support outside \(C\).  Hence the CBF left side is zero.
Moreover,

\[
|v_i|\le v_f\le v_{\max},\qquad
|\ell\omega_i|\le v_f\le\ell\omega_{\max},
\]

and \(q_C^\top Bz_f=mv_f\), proving feasibility of every row.
The optimizer therefore satisfies
\(\dot s_C\ge v_{\mathcal P}^{\rm uni}\) and exits within
\(L/v_{\mathcal P}^{\rm uni}\) under continuous evaluation.

For a native input held for \(\delta\), define

\[
L_u=\omega_{\max}
\sqrt{v_{\max}^2+(\ell\omega_{\max})^2}.
\]

Since
\(\|\frac{d}{dt}(B_i(\theta_i)z_i)\|\le L_u\), each complete held step obeys

\[
s_C(t_{k+1})-s_C(t_k)
\ge\delta\left(v_{\mathcal P}^{\rm uni}-\frac12L_u\delta\right).
\tag{30c}
\]

Thus finite sampled exit follows whenever
\(v_{\mathcal P}^{\rm uni}>\frac12L_u\delta\).

### Corollary 5b (bounded-input unicycle arrival transfer)

If a virtual execution enters and retains \(B(q_i,r_y)\) in finite time, then

\[
\|x_i-q_i\|
\le\|y_i-q_i\|+\|x_i-y_i\|
\le r_y+\ell.
\tag{30d}
\]

Thus any already-proved retained virtual arrival transfers to the physical
ball \(B(q_i,r_x)\) whenever \(r_x\ge r_y+\ell\).  This implication does not
prove that virtual CLEAR is globally live; it transfers a liveness guarantee
only where one is already available, such as Theorem 5 or Corollary 5a.

---

## 8. What the static certificate does not prove

Theorem 3 excludes \(u^*=0\) at certified states.  It does not exclude:

- a periodic orbit;
- recurrent obstacle following;
- a phase-locked orbit around parked robots;
- Zeno switching of the active constraint graph; or
- convergence to a non-goal invariant set with nonzero velocity.

This boundary is essential.  A norm or moment margin can certify motion without
certifying mission progress.

Define

\[
V(x)=\frac12\sum_i\|x_i-q_i\|^2.
\tag{31}
\]

Let \(\tau_k\) be the ordered union of congestion exits and fixed-duration
checkpoints.  Let \(A_k\) be the set of robots not yet retained at their goals.

### Theorem 6 (conditional event-sampled arrival)

Assume:

1. sample times have a positive lower spacing;
2. outside congestion, \(\dot V\le-\alpha\sum_{i\in A_k}\|x_i-q_i\|^2\);
3. at every congestion transition, either the arrived set strictly grows or
   \[
   V(\tau_{k+1})\le V(\tau_k)-\delta
   \tag{32}
   \]
   for a uniform \(\delta>0\);
4. arrived robots are retained; and
5. all trajectories remain in a compact subset of \(\mathcal S\).

Then all robots enter and remain in their arrival balls after finitely many
congestion events and finite time.

**Proof.**  Because \(V\ge0\), (32) can occur only finitely many times before
the arrived set grows.  That set can grow at most \(N\) times.  Hence only
finitely many congestion transitions occur without complete arrival.  On the
remaining free-flow intervals, assumption 2 gives convergence to the goal
balls; positive event spacing rules out accumulation in finite time, and
assumption 4 gives retention. \(\square\)

Theorem 6 is deliberately conditional.  Its antecedents must be checked
transition by transition; they are not consequences of Theorem 3.

---

## 9. Direct implementation correspondence

| Mathematical object | Implementation |
|---|---|
| \(g\), Eq. (6) | `CLEARController.goal_field` |
| raw \(c\), Eqs. (7)--(8) | `circulation_field` |
| \(A_T\), Eq. (12) | `_geometric_rows(..., tangent_only=True)` |
| \(z_{\rm base}=\Pi_{\mathcal K_T}c\), Eq. (13) | `project_halfspaces` |
| rigid component and token, Eqs. (14a)--(14c) | `cluster_escape_field` and `_cluster_tokens` |
| optional adaptive gain, Eq. (20a) | `_circulation_components` and `command` |
| full CBF set, Eq. (16) | `_geometric_rows(..., tangent_only=False)` |
| CBF projection, Eq. (17a) | `project_halfspaces` |
| safe actuator scaling, Eq. (17b) | common scale in `command` |
| unicycle realization, Eqs. (17c)--(17f) | `unicycle.simulate_unicycle` |
| certified native bridge row, Eqs. (30a)--(30c) | `_certified_bridge_progress` and `UnicycleRollout.certified_bridge_*` |
| global witness margin, Eq. (18) | `ControllerAudit.global_tangent_margin` |
| pre-scale projected command | `ControllerAudit.projected_command` |
| straight-bridge runtime domain audit | `theorem_audit.audit_straight_bridge_tokens` |
| solver audits | `cbf_minimum_residual`, `tangent_fallback_steps` |
| inter-sample audit | swept pair distance and two obstacle subsamples |

The unit tests verify pair reciprocity and tangency, permutation equivariance,
rotation equivariance, the reflection--chirality relation, contact safety, exact
speed bounds, the global witness identity, virtual-clearance inflation, and a
deterministic two-robot symmetric-swap escape.

---

## 10. Required falsification sequence

Before treating CLEAR as a paper algorithm, run the following in order.

1. **Algebraic tests:** all tests in `tests/test_core.py` must pass at numerical
   tolerance.
2. **Exact symmetric probes:** HeadOn-2, Cross-4, and antipodal Swap
   \(N=8,20,40\), with zero jitter.
3. **Ablations:**
   - \(\kappa=0\) (CBF + goal only);
   - raw circulation without (13);
   - pair circulation only;
   - boundary circulation only;
   - nullspace equality \(A_Tz=0\) instead of the one-sided cone.
4. **Certificate audit:** report the fraction of congested steps with
   \(\Delta_T>0\), every tangent fallback, every CBF residual, and swept
   clearance.
5. **Livelock audit:** for every failure, detect recurrent states and evaluate
   event inequality (26).  Do not relabel a moving periodic orbit as a
   deadlock.
6. **Benchmark:** use the same generated instances and 60 s timeout for CLEAR,
   ORCA, MGR, and other baselines.  Published aggregates must remain visibly
   separate from locally reconstructed results.

The decisive hypothesis is not merely “circulation helps.”  It is:

> Projecting a unified robot/boundary circulation into the one-sided safety
> tangent cone improves dense-clutter liveness while preserving exact safety,
> and does so more reliably than either raw circulation or equality-nullspace
> circulation.

If the one-sided-cone ablation does not beat both alternatives on paired
instances, the proposed structural contribution is falsified.
