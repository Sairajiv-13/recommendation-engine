# ADR 0001: TruncatedSVD for matrix factorization, not a deep model

Status: accepted.

## Context

The catalog here is MovieLens latest-small: about 100,000 ratings, 610
users, and a few thousand movies with at least one rating. The model has
to train on a laptop, retrain cheaply in CI-adjacent jobs, and serve from
a small container. A neural approach (two-tower, autoencoder, or a
sequential model) is the fashionable default for recommendations, so the
choice needs a stated reason.

## Decision

Use classical factorization, TruncatedSVD over the user-mean-centered
rating matrix, as the learning-based model, and skip deep architectures
at this scale. Keep a popularity baseline and an item-item cosine model
next to it, evaluated on the same holdout, and let NDCG at 10 decide
which model the API serves.

Outcome on the first full run: item-item cosine won (NDCG at 10 of
0.2237 against 0.1660 for SVD and 0.1561 for popularity), so item-item
is the serving model and SVD stays in the results table as the
comparison point. The case against a deep model below stands either
way, and the gap is a reminder that the fancier model is not
automatically the better one on small data.

## Reasons

- Data size. A few hundred users cannot support a deep model with
  thousands of embedding parameters per user and item without heavy
  regularization and careful tuning. SVD has one real knob, the number
  of components, and it is stable at this scale.
- Training cost. The SVD fits in seconds on CPU. A deep model would need
  a training loop, negative sampling, and early stopping logic, all to
  chase a gain that the holdout cannot reliably measure on 610 users.
- Serving cost. Prediction is a row lookup in a precomputed matrix. No
  model server, no GPU, no batching questions.
- Debuggability. When a recommendation looks wrong, the user mean, the
  latent factors, and the seen-item filter are all inspectable. That
  matters more for a portfolio-grade system than a small metric gain.

## Trade-offs accepted

- SVD reads ratings as numbers on a shared scale. It cannot model
  sequence, context, or side features such as genres except through the
  separate cold-start path.
- The full predicted matrix is materialized, which is fine at this
  catalog size and would not be fine at a million users. At that scale
  the serving layer would switch to factor lookup and dot products per
  request, and a two-tower model would be worth revisiting.
- TruncatedSVD treats missing entries as zero after centering. Implicit
  feedback methods such as ALS with confidence weighting handle the
  missing-as-negative question differently and are the first alternative
  to try if this system grows.
