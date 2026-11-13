# UGP Parts I and II: speaker notes

Slides 1–33 are the talk, about 30 seconds each (11 to 12 minutes at a calm pace); slides 34–38 are backup for questions.

## 1. Teaching Neural Networks to Forecast Fluid Flow

Good morning, sir. My UGP had two parts, both about one idea: teaching a neural network to forecast how a fluid flow moves. Part I was an international competition on a real airfoil wake. Part II was a controlled study that answers the question Part I left open: what does pretraining data actually buy?

## 2. A neural network can forecast a flow in a blink; we asked how to build one well

A neural forecaster takes the flow now and just before, and predicts the flow a moment later; feeding its predictions back in gives longer forecasts. On the left, Part I: a real wake measured in a water tunnel, noisy and limited. On the right, Part II: a flow I simulate, where I control the data and always know the exact answer.

## 3. Part I: Forecasting a Real Airfoil Wake

Part I: the NeurIPS 2026 RealPDE competition, Track 1, where the challenge was to forecast real measured flow.

## 4. Forecasting a real wake is hard: noisy data, unseen conditions, strict rules

This is the data: velocity behind a NACA 4418 airfoil, measured with particle image velocimetry. Blue is the slow wake, red the fast flow around it. The model sees 20 frames and predicts the next 20, with a band around every value. We had 81 recordings, a 256-megabyte limit and one scored submission a day, and test conditions go beyond the training range.

## 5. Uncertainty was worth a quarter of the score; we recovered the hidden weights

The score mixes five channels, but the weights were never published. I built a small tool that read the public leaderboard and solved for the weights from 40 published scores. It matched every total to within 0.005 and showed that uncertainty alone was worth almost a quarter of the score. So it became our focus.

## 6. Only narrow bands that still contain the truth earn uncertainty points

Here is how the uncertainty channel works. For each value we give a band. If the truth is inside, the score is e to the minus width over sigma, so narrower is better. If the truth is outside, the score is zero. A narrow, correct band scores 0.84, a wide one 0.17, and a narrow miss scores nothing.

## 7. Two paths: a fine-tuned forecast, and uncertainty bands that never touch it

Our system has two paths. The top path makes the forecast: the organisers' Fourier neural operator, fine-tuned on the real data, with several copies averaged into one model soup, plus a small network that restores the turbulence smooth forecasts lose. The bottom path makes the bands: a U-Net predicts how wrong the forecast is at each point, and a 24-bin table turns that into a width. The bands never touch the forecast, so accuracy can never get worse.

## 8. The entry raised the overall score from 72.7 to 80.1; uncertainty nearly tripled

Against the organisers' baseline, the overall score rose from 72.7 to 80.1. The biggest gain is uncertainty, plus 25 points, and turbulence rose 4.6. Accuracy and speed stayed at the baseline's high level.

## 9. The score climbed in small steps; adding uncertainty bands gave the largest jump

This is every submission that beat the best so far, over one month. The first big step is adding the uncertainty bands. Then the turbulence fix, and finally calibrating the band widths, which took us across our target of 80.

## 10. We finished just outside the top 50; the top ten needed a better forecast, not better bands

Where we stand. The field was very tight: 80.08 sat just outside the top 50 of about 182 teams. One more insight: even perfect uncertainty bands on our forecast would score 81.73, below the top-10 cutoff. So the top ten needed a better forecast, not better bands.

## 11. What failed was just as informative: a 2-D physics rule is false for this data

We also measured what did not work. Forcing the forecast to be divergence-free cost nine points: the camera sees a 2-D slice of a 3-D flow, so the 2-D rule is not true for this data. The benchmark's best architecture, a U-Net, was 16 percent worse. Adding simulations to training gave nothing. And a finer band table changed almost nothing.

## 12. Part II: What Does Pretraining Data Buy?

That left one question open: would pretraining on lots of related data have helped? Part II answers it on a flow we control.

## 13. Research question: which property of pretraining data produces the gain?

Large models are pretrained on lots of related data, then fine-tuned on the task, instead of training from scratch. Aurora, Microsoft's weather model, reports a 25 percent gain from pretraining. But its data differ in amount, variety and closeness all at once, so it cannot say which one helps. That is our research question.

## 14. Kolmogorov flow: a turbulent flow whose every property we control

We chose Kolmogorov flow. In one line: the spin is carried by the flow, smoothed by viscosity, damped by drag, and kept alive by a striped forcing. Two knobs set its behaviour: the number of stirring bands, and the Reynolds number, how wild it gets. Our target is Reynolds 300 with four bands, never seen in pretraining.

## 15. Pretraining sets differ in size and variety; only the largest includes near-target settings

This is the design of the main experiment. Each dot is a flow setting. The pretraining sets have 2, 8 or 32 settings, at three different amounts of data. Only the 32-setting set contains the target's close neighbours, same bands and nearby Reynolds numbers. Keep this map in mind for the results.

## 16. We built a verified toolkit, and pre-registered every experiment

I built four pieces, each tested before use. A simulator whose energy balance is exact to one part in a billion. A dataset of 326 simulations. Aurora rebuilt in JAX, 5.1 million parameters. And a strict protocol: every experiment's predictions were written down and committed before running. In the original experiments, 25 of 49 predictions hit; the misses are findings too.

## 17. Inside the model: patches, windowed attention, and a zoom-out, zoom-in backbone

Inside the model: two frames go in, cut into 256 patches. The Perceiver compresses the frequency levels, and a Swin U-Net lets patches share information in small windows, zooming out and back in. It outputs the next snapshot. The core block matches Aurora's own code to seven decimals, and cheap LoRA fine-tuning adds a small low-rank update, training under one percent of the weights.

## 18. Eleven experiments, one question each

Eleven experiments, each answering one question; here is the map, with the one-line answer for each. All of them use two scores: the relative error of a forecast, and the gain, which is how much lower the pretrained model's error is than the scratch model's.

## 19. Pretraining roughly halves the next-step error, from 0.150 to 0.081

First result: pretraining roughly halves the next-step error, from 0.150 to 0.081. You can see it in the flow: from the same starting moment, the pretrained forecast keeps the vortices that the scratch model smears. It holds on flows nearly three times wilder than any it was pretrained on.

## 20. Pretrained models reach in 100 steps what scratch never reaches in 22,000

And it is fast. Trained from scratch, the model needs 22,000 steps to reach 0.150. Both pretrained models reach 0.08 within 100 fine-tuning steps. The one pretrained on close settings is already there before any fine-tuning.

## 21. Closeness to the target matters most: near-target settings give the best start

Second result, the main answer: closeness matters most. More data or more variety did not change the final gain. But the set containing near-target settings starts far better, before any fine-tuning: error 0.084, against about 0.19.

## 22. Pretraining pays most when target data is scarce: 41% gain with one simulation

Third: pretraining pays most when target data is scarce. With one simulation of the target, the gain is 41 percent; with six, training from scratch nearly catches up. That is exactly the situation of real experiments.

## 23. Long forecasts are hard: small errors grow until the fine detail is lost

Forecasting far ahead is hard. Feed the prediction back in, and small errors grow: by step 10 the fine detail is gone, and by step 20 the forecast no longer matches the truth, even though the big vortices are still there. So we also measured 20- and 50-step forecasts.

## 24. Pushforward is the biggest lever for long forecasts; attention and depth help next steps

Fourth: what improves the model. Training on its own predictions, pushforward, cuts long-forecast drift by 31 to 50 percent, the biggest lever. Two attention upgrades give about 9 percent, a deeper model about 7, and training became twice as fast. The Perceiver did not earn its place: simple padding beat it.

## 25. On real data, the rebuilt model is within one point of the organisers' model

Fifth, back to the real wake. Our rebuilt Aurora scores 94.7 on accuracy, within one point of the organisers' model while seeing two past frames instead of twenty. Simulation pretraining adds only about 0.1. On real data, the real data is what counts, the same lesson as Part I.

## 26. The remaining error lives in the tiniest swirls; pretraining pushes that limit further

Then I asked what still limits accuracy, without new training. Each row is a group of models; each column a swirl size, big on the left. Big swirls are almost exact. The error grows for smaller swirls until it is as large as the signal, the white bar. Pretraining pushes that limit to smaller swirls.

## 27. Two thirds of the error is in the spin field, and scratch memorises its data (32×)

Two more findings. About two thirds of the error sits in the spin field, the one richest in small swirls. And the scratch model is 32 times better on its training data than on new data: it memorises, so it needs data, not longer training. Pretrained models do not memorise.

## 28. On this flow the spin fixes everything else, so the model can predict the spin alone

That suggested a fix. On this periodic box, the spin determines everything else exactly: velocity comes from the stream function, and pressure from the velocity. So the model predicts only the spin, and we calculate the rest.

## 29. The normal model breaks the no-leak rule by 12%; with physics built in, it never does

Here is why it matters. Real flow cannot create or destroy fluid; this map shows where a prediction does. The normal model breaks that rule by 12 percent, speckled everywhere. With the physics built in, the rule holds exactly, just like the true flow.

## 30. Building the physics in cuts the average 20-step forecast error by 13%

And it forecasts better. The average error over a 20-step forecast falls by 13 percent, the best of any change I tried on this task, pushforward included. Training is 7 percent faster and pressure error falls 38 percent.

## 31. Together: closeness, scarce target data and true physics are what help

Putting both parts together. Pretraining helps when it is close to the task and target data is scarce; far-away simulations did not help on the real wake. Physics helps when it is true for the data: it hurt on camera measurements and helped on the simulation. And writing predictions down first turns every surprise into a result.

## 32. Limits and next steps: claims are bounded by a small model and one flow family

To be clear about the limits: a small model, one flow family, fixed budgets and three seeds, so these are claims about this setting, not about Aurora itself. Next: combine the physics-built-in model with pushforward and the attention upgrades, weight training toward small swirls, and give the model a longer, ordered history.

## 33. Conclusions: pretraining helps when close and scarce; built-in physics steadies forecasts

To conclude. Part I raised the organisers' model from 72.7 to 80.1 by decoding the score and calibrating uncertainty. Part II shows that pretraining halves the error and helps most when close and scarce, and that building physics into the output steadies forecasts. The code, reports and every result are on GitHub. Thank you, sir; I am happy to take questions.

## 34. References

Backup: the references for the sources cited on the slides.

## 35. Appendix A: how the competition score is computed

Backup: the formulas behind the competition score, recovered by least squares from 40 public leaderboard rows.

## 36. Appendix B: from the predicted spin to velocity and pressure

Backup: the exact formulas for deriving velocity and pressure from the spin, and the three variants compared.

## 37. Appendix C: the prediction record

Backup: the full prediction record. Of 49 pre-registered predictions in the original experiments, 25 hit; the two later experiments add 3 of 6 each.

## 38. Appendix D: glossary

Backup: a short glossary of the terms used in the talk.
