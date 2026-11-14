# UGP Parts I and II: speaker notes

One deck: Part I (UGP-I) and Part II (UGP-II). About 30 seconds a slide.

## 1. Title

Good morning, sir. My UGP had two parts. In Part I, I forecast a real airfoil wake in the NeurIPS RealPDE competition. In Part II, I rebuilt Microsoft's Aurora model to answer the question Part I left open: what does pretraining data actually buy?

## 2. What is Covered

The plan: first the competition, its problem, our approach and the result. Then the question it left open, and what Part II found.

## 3. Part I cover: Forecasting Real Airfoil Wakes

Part I: forecasting real airfoil wakes in the NeurIPS 2026 RealPDE competition.

## 4. This Is the Data

Walk them through the picture. NACA4418 airfoil, 20 degrees angle of attack, Re 13,950. Blue is slow fluid, red is fast. The dark blue lobe is the separated wake. Note the speckle — that is genuine measurement noise from the PIV cross-correlation, not a rendering artefact. The task is: see the top three frames (and 17 others), produce the bottom three (and 17 others). Twenty frames in, twenty frames out. Mention that in the real competition the test runs are at Reynolds numbers we never trained on.

## 5. The Task, in a Nutshell

Keep this short. The one number that matters is the last line: the test set extrapolates beyond the training range in Reynolds number, so this is not interpolation. Pressure is in the tensor but identically zero — PIV cannot measure it.

## 6. What We Actually Submit

Make the deliverable concrete before talking about methods. We submit a zip containing model weights plus a Python file exposing one function, predict(). The graders run it inside an isolated container with no internet, PyTorch and NumPy only, and a 5-minute budget. It returns three arrays: the forecast, and a lower and upper bound for every single value. The size bar shows why everything is tight: the forecasting model alone eats 201 of the 256 MB, which is why we could never simply ensemble several models.

## 7. Why This Is Hard

On the third point — the plot shows one row of one frame. The jitter is measurement noise. No model can predict that, so a floor on achievable error exists. This becomes important later: it is why chasing accuracy forever does not pay, and why the uncertainty channel is where the value is.

## 8. The Rules

This slide explains the whole working style. With 24 hours between measurements you cannot iterate by trial and error — you must predict, submit, then reconcile. Most of what follows is the consequence of that constraint.

## 9. How a Submission Is Scored

The organisers publish the five channel scores but not how they combine into the final number. We solved for the weights by least squares across 40 leaderboard rows and got a fit accurate to 0.005. That is what let us decide where to spend effort — and it showed the uncertainty channel, which we had been ignoring, was worth 24.7%.

## 10. The Unusual Channel: Uncertainty

Work through the four rows. Key consequence for later: because the reward decays exponentially with width, this channel is set by the TYPICAL error, whereas the accuracy channel is an L2 norm dominated by the WORST errors. They are nearly independent objectives — a model can be no more accurate overall and still win this channel outright.

## 11. The Tool built

Be upfront about the legality question because someone will ask. Codabench serves the leaderboard to the browser through a public REST endpoint, and at the time it exposed per-channel subscores for every submission. We read that endpoint programmatically instead of clicking through the website. No authentication, no private data, nothing hidden. The competition rules forbid network calls from inside the submitted code during evaluation — that is a completely different thing, and our submission makes none. Worth noting the endpoint has since been restricted, so only totals are public now.

## 12. Approach 1 of 3:  The Model

Fine-tuning was the single biggest step — it took us from roughly 77 to 78.45. Model soup means training several times from different random starts and averaging the weights; it cancels each run's idiosyncrasies. The TKE head was worth +2.21 on that channel alone. But flag the caveat now and pay it off on slide 18 — this is the thread of the talk.

## 13. Approach 2 of 3:  Uncertainty

The green box is the important one for a sceptical audience: because the prediction array is untouched, these submissions carry zero accuracy risk. We verified that byte-for-byte before shipping each one. Every gain since 21 September came from this channel with the model completely frozen.

## 14. Approach 3 of 3:  Speed

Start with the curve, because the formula is unintuitive. The score is 100/(1 + sqrt(t/728.96)). It is deliberately gentle: taking 2.5 minutes still scores 65, and the hard failure is only at 5 minutes. We run in 7.8 seconds, which scores 90.6 — we are far out on the flat part, so further speed-ups buy almost nothing. That is the point of the slide: we did the cheap engineering (GPU, merged networks, half precision), confirmed we were in the flat region, and stopped. One tempting idea — cutting the model's internal padding — gave a genuine 1.74x speed-up and wrecked accuracy by 60 times more than the time gain was worth. We priced it before submitting rather than after.

## 15. Inside Each Submission

This is the 'specific steps' table. Three rows carry most of the story, highlighted in orange. SHIFT_v2 is where the uncertainty machinery was added at all — the largest single jump in the project. FA1BMT added the TKE head. NARROW080 is the final one, and it is the cleanest illustration of the whole approach: the forecasting model is completely frozen and byte-identical, we only multiplied every interval width by 0.80, and that was worth +0.20. Note the 'model underneath' column: after FITA1B the backbone stops changing entirely. Everything after that is uncertainty and engineering.

## 16. The Steps That Moved the Score

Point out three places: SHIFT_v2 is where the uncertainty channel was added — the largest single jump. FA1BMT is the TKE head. The final two points are pure bounds calibration with the model frozen. Also note the flat stretch in the middle: that is six weeks of accuracy work producing almost nothing, which is the puzzle slide 18 resolves.

## 17. Getting It Wrong, Then Right

This is the slide that front-runs the obvious question, so do not skip it. The failed submission is what let us solve for the calibration constant — one equation, one unknown. The answer was 1.00, not 2.14, meaning the intervals should have been narrowed all along. Root cause: an earlier successful change had altered interval shape and width together and we credited the gain to width. The next day's submission, built on the corrected constant, crossed 80.

## 18. Where We Stand

Be straightforward here. We are at 80.06, rank 1 is 82.10, the top-10 cutoff is 81.74. In relative terms that is a 2.5% difference, but the field is so tightly packed that 2.5% spans rank 1 to outside the top 50. The left panel is there to make the scale honest — every team is inside one dot on the full range.

## 19. What the Gap Consists Of

Explain the oracle: for a given set of forecast errors there is a best possible set of intervals — knowing each error exactly and sizing the interval to just cover it. That is an upper bound no method can beat. We reach 83.6% of it; the leader reaches 90.7%. This is how we separated 'they have a better model' from 'they have better uncertainty' using only published numbers.

## 20. Main Finding

Each point is one past submission: its offline score against the score it actually got. For a useful test these should lie on a rising line. For rel_l2 they do not — the fit is negative and meaningless, because our offline score spans 0.028 while the live score spans 0.098. The test has no resolution in that channel. The uncertainty panel, by contrast, is clean: slope 0.62, and that matches a factor we derived independently from a single live anchor. This is the result I would most like feedback on.

## 21. Part I: Where We Landed

Where Part I landed. Our development score was 80.08, against the organisers' baseline of 72.65. The top-10 cutoff was 81.90 and the leader 82.22. Even perfect uncertainty bands would not reach the top ten, so the forecast itself was the limit. We finished just outside the top 50 in a very tight field, and the project left one question open: would pretraining on related data have helped?

## 22. Part II cover: What Does Pretraining Really Buy?

Part II answers that question: what does pretraining data really buy? I rebuilt Aurora, a large weather model, from scratch in JAX and tested it on a swirling flow where I control the data.

## 23. Where We Started

In UGP-I, I took part in the RealPDE competition at NeurIPS 2026, forecasting real measured airflow past a wing, and scored 80.08. It ended with an open question: would a model pretrained on lots of related data have done better? I could not test that, because outside pretrained models were banned and there were only 81 real recordings. UGP-II builds a setting where I can test it properly.

## 24. The Big Idea: Pretrain, Then Fine-tune

Quick background on how these models are trained. First, pretraining: learn from lots of related data. Then a short fine-tuning on the actual task. Think of a student who studies every chapter, then practises the exam chapter. Training from scratch is studying only the exam chapter. Throughout the project, I compare these two.

## 25. Aurora Says It Works, But Not Why

Aurora is the best-known example. It has 1.3 billion parameters and reports that pretraining beats training from scratch by 25%. But its pretraining data differs in amount, variety, format and closeness to the task all at once, so it cannot say which one helps. Rerunning that comparison would need a lot of GPUs. That is the gap I target.

## 26. Our Goal

So my goal has two parts. Rebuild Aurora from scratch in JAX, then use it to find out which property of the pretraining data creates the benefit. Four objectives: rebuild Aurora at small scale, build my own data solver, test each ingredient one at a time, and finally check the answer on the real lab data from UGP-I.

## 27. Why Kolmogorov Flow?

I chose Kolmogorov flow, a classic test case: fluid in a box, kept moving by a steady push. The big advantage is that I generate the data myself, in any amount and any setting. Two simple knobs change how it behaves. It is genuinely turbulent, so forecasting it is hard. And I always know the exact true answer to score against.

## 28. Two Knobs Define Each Flow

These are the two knobs. On the left is the stirring pattern, kf: bands of fluid pushed left and right, and kf is the number of band pairs. On the right is the Reynolds number: how wild the flow gets, from calm at 100 to turbulent at 1000. My test task is Reynolds number 300 with kf 4, and the model never sees that setting during pretraining.

## 29. The Task: Predict the Next Moment

The model's job is simple. It sees the flow at two consecutive moments and predicts the next one; two inputs let it see the motion. For longer forecasts, each prediction is fed back in as the next input, step after step. I test up to 20 steps ahead.

## 30. How We Measure Success

I use four measures. Error: how far the prediction is from the truth. The lazy guess, 'nothing changes', is the minimum bar any model must beat. Long-forecast drift measures stability over 20 steps. And the key number, pretraining gain: how much lower the error is than a model trained from scratch on the same data and budget.

## 31. What We Built

I built four pieces, each tested before use. A fluid simulator whose energy balance is exact to about one part in a billion. A dataset of 326 simulations across 33 flow settings. A copy of Aurora with 5.1 million parameters that matches Aurora's own code to seven decimal places. And an experiment system where one settings file gives one run and one recorded result.

## 32. Inside the Model

Inside, the model uses Aurora's building blocks. It cuts each snapshot into 256 small patches. Each patch gathers information from the patches that matter to it; this is called attention, done in small windows for speed. Then the model zooms out to see the big swirls and back in to keep the detail, and outputs the next snapshot.

## 33. Result 1: Pretraining Halves the Error

First result: pretraining halves the error. A model trained from scratch has an error of 0.150; the pretrained one gets 0.080, which is 42 to 50% better in every setup tested. On the right, the colours show the fluid's spin. Starting from the same moment, the pretrained model's next-step forecast is closer to the truth than the scratch model's.

## 34. Result 2: Closeness Matters Most

Second, and the main answer to my question: closeness matters most. Adding more data or more variety did not change the gain. But the pretraining mix that contained settings near the task gave a far better starting point, 0.085 against about 0.19. On the right: even when tested well beyond the training range, pretraining still gives a 36 to 41% gain.

## 35. Result 3: Most Valuable When Data Is Scarce

Third: pretraining is most valuable when task data is scarce. With a single target simulation it gives a 41% gain, and as more target data becomes available, training from scratch catches up. Even without any fine-tuning, the pretrained model is already within 6% of the best fine-tuned result.

## 36. Result 4: What Makes the Model Better

Fourth: what makes the model better. The biggest win is pushforward, training the model on its own predictions, which cuts long-forecast drift by up to half. Two small attention upgrades give about 9% at very little cost, and a deeper model about 7%. Cheap fine-tuning, called LoRA, corrects systematic errors: up to 71% in one case.

## 37. Result 5: Tested on Real Lab Data

Finally, I went back to the real UGP-I airflow data. On top: my UGP-I entry scored 80.1 overall, against 72.7 for the organisers' baseline. Below, on held-out real recordings, my rebuilt Aurora scores 94.7 accuracy, within one point of the organisers' FNO, while seeing only 2 past frames instead of 20. Simulation pretraining adds under 0.1, so on real data, the real data is what counts.

## 38. What Still Limits Accuracy

After the review, I examined the 30 trained models to find what still limits them, without any new training. Three things. The big swirls are almost perfect; the error lives in the tiniest swirls. About 70% of the error is in the spin field, which has the most tiny swirls. And the model trained from scratch memorises its two examples, so longer training cannot fix it; the pretrained models do not memorise.

## 39. Key Takeaways

To sum up: pretraining halves next-step error. The main answer to my question is that closeness to the task matters more than amount or variety. The payoff is biggest when task data is scarce. Training on its own predictions keeps long forecasts steady. And building the physics into the model keeps 20-step forecasts 13 percent steadier.

## 40. Future Work

There are four ways to take this further. First, build the physics in. I already ran the first test: the model predicts only the spin, and the other fields are worked out from it exactly. The flow's rules then always hold, and 20-step forecasts improve by 13%. Next is combining it with pushforward. Then steadier long forecasts, a longer and ordered input history, and scaling up with better transfer to real measurements.

## 41. Both Parts Together

Putting both parts together. Know what you are measured on. Pretraining helps when it is close to the task and target data is scarce, and far-away simulations did not help on the real wake. Physics helps only when it is true for the data: it hurt on the camera data in Part I and helped on the simulation in Part II. And writing predictions down first turns every surprise into a result.

## 42. Appendix

The details are in the final report's appendices: the scoring formulas, the model and data, all eleven experiments with the prediction record, and the physics derivation. I am happy to go into any of them.

## 43. Thank You

Thank you, sir. I am happy to take questions.
